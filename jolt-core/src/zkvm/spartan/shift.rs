use std::sync::Arc;

use allocative::Allocative;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPlusOnePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::{
    evaluate_shift_sumcheck_witnesses, generate_shift_sumcheck_witnesses,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

// Spartan PC sumcheck
//
// Proves the batched identity over cycles j:
//   Σ_j EqPlusOne(r_cycle, j) ⋅ (UnexpandedPC_shift(j) + γ·PC_shift(j) + γ²·IsNoop_shift(j))
//   = NextUnexpandedPC(r_cycle) + γ·NextPC(r_cycle) + γ²·NextIsNoop(r_cycle),
//
// where:
// - EqPlusOne(r_cycle, j): MLE of the function that,
//     on (i,j) returns 1 iff i = j + 1; no wrap-around at j = 2^{log T} − 1
// - UnexpandedPC_shift(j), PC_shift(j), IsNoop_shift(j):
//     SpartanShift MLEs encoding f(j+1) aligned at cycle j
// - NextUnexpandedPC(r_cycle), NextPC(r_cycle), NextIsNoop(r_cycle)
//     are claims from Spartan outer sumcheck
// - γ: batching scalar drawn from the transcript

/// Degree bound of the sumcheck round polynomials in [`ShiftSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

/// Sumcheck prover for [`ShiftSumcheckVerifier`].
#[derive(Allocative)]
pub struct ShiftSumcheckProver<F: JoltField> {
    combined_witness_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_r_cycle: MultilinearPolynomial<F>,
    eq_plus_one_r_product: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> ShiftSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheck::new_prover")]
    pub fn gen(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let params = ShiftSumcheckParams::new(state_manager, key);

        let (preprocessing, _, _program_io, _final_memory_state) = state_manager.get_prover_data();
        let trace = state_manager.get_trace_arc();

        let (_, eq_plus_one_r_cycle) = EqPlusOnePolynomial::<F>::evals(&params.r_cycle.r, None);
        let (_, eq_plus_one_r_product) = EqPlusOnePolynomial::<F>::evals(&params.r_product.r, None);

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (combined_witness_poly, is_noop_poly) =
            generate_shift_sumcheck_witnesses(&preprocessing.shared, &trace, &params.gamma_powers);

        Self {
            combined_witness_poly,
            is_noop_poly,
            eq_plus_one_r_cycle: eq_plus_one_r_cycle.into(),
            eq_plus_one_r_product: eq_plus_one_r_product.into(),
            bytecode_preprocessing: preprocessing.shared.bytecode.clone(), // HACK
            trace,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftSumcheckProver<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let univariate_poly_evals: [F; DEGREE_BOUND] = (0..self.combined_witness_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let combined_witness_evals = self
                    .combined_witness_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let eq_r_cycle_evals = self
                    .eq_plus_one_r_cycle
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let eq_r_product_evals = self
                    .eq_plus_one_r_product
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let is_noop_evals = self
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);

                std::array::from_fn(|i| {
                    combined_witness_evals[i] * eq_r_cycle_evals[i]
                        + self.params.gamma_powers[4]
                            * (F::one() - is_noop_evals[i])
                            * eq_r_product_evals[i]
                })
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut running, new| {
                    for i in 0..DEGREE_BOUND {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        rayon::scope(|s| {
            s.spawn(|_| {
                self.combined_witness_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                self.is_noop_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                self.eq_plus_one_r_cycle
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
            s.spawn(|_| {
                self.eq_plus_one_r_product
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            });
        });
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = get_opening_point::<F>(sumcheck_challenges);
        let is_noop_eval = self.is_noop_poly.final_sumcheck_claim();

        let [unexpanded_pc_eval, pc_eval, is_virtual_eval, is_first_in_sequence_eval] =
            evaluate_shift_sumcheck_witnesses(
                &self.bytecode_preprocessing,
                &self.trace,
                &opening_point,
            );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_virtual_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_first_in_sequence_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ShiftSumcheckVerifier<F: JoltField> {
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> ShiftSumcheckVerifier<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let params = ShiftSumcheckParams::new(state_manager, key);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_virtual_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let r = get_opening_point::<F>(sumcheck_challenges);
        let eq_plus_one_r_cycle_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_cycle.r.to_vec()).evaluate(&r.r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_product.r.to_vec()).evaluate(&r.r);

        [
            unexpanded_pc_claim,
            pc_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .iter()
        .zip(&self.params.gamma_powers)
        .map(|(eval, gamma)| *gamma * eval)
        .sum::<F>()
            * eq_plus_one_r_cycle_at_shift
            + self.params.gamma_powers[4]
                * (F::one() - is_noop_claim)
                * eq_plus_one_r_product_at_shift
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = get_opening_point::<F>(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }
}

struct ShiftSumcheckParams<F: JoltField> {
    gamma_powers: [F; 5],
    n_cycle_vars: usize, // = log(T)
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    r_product: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ShiftSumcheckParams<F> {
    fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let gamma_powers = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5)
            .try_into()
            .unwrap();

        let n_cycle_vars = key.num_steps.ilog2() as usize;
        let (outer_sumcheck_r, _) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(n_cycle_vars);
        let (product_sumcheck_r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );
        let (r_product, _) = product_sumcheck_r.split_at(n_cycle_vars);

        Self {
            gamma_powers,
            n_cycle_vars,
            r_cycle,
            r_product,
        }
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, input_claim_next_pc) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, input_claim_next_unexpanded_pc) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_virtual) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_first_in_sequence) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
        let (_, input_claim_next_is_noop) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );

        input_claim_next_unexpanded_pc
            + input_claim_next_pc * self.gamma_powers[1]
            + input_claim_next_is_virtual * self.gamma_powers[2]
            + input_claim_next_is_first_in_sequence * self.gamma_powers[3]
            + (F::one() - input_claim_next_is_noop) * self.gamma_powers[4]
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
