# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# This software and associated documentation files contain proprietary and
# confidential information of INFIDOS LLP and BALAVIGNESH M. Unauthorized
# copying, modification, distribution, transmission, or reproduction of this
# material, via any medium, is strictly prohibited without prior written
# permission from INFIDOS LLP.
# ==============================================================================

"""
Mayon-Cortex Reasoning — Multi-Hop Proofs, Logic, Math & Puzzle Solving
======================================================================
Path ranking, symbolic logic, System 2 thinking, ARC visual solver,
symbolic algebra, number theory, linear algebra, calculus, and universal puzzle solvers.
"""

from mayon_cortex.reasoning.path_ranker import PathRanker, ScoredPath
from mayon_cortex.reasoning.logic import LogicEngine, ProofStep, FormalProof
from mayon_cortex.reasoning.thinking import ThinkingEngine, ThoughtResult
from mayon_cortex.reasoning.problem_solver import ProblemSolver, ProblemDecomposition, SolutionStep, SolvedProblem
from mayon_cortex.reasoning.judgment_engine import JudgmentEngine, JudgmentVerdict
from mayon_cortex.reasoning.analogy import AnalogyEngine, AnalogyResult
from mayon_cortex.reasoning.cross_domain import CrossDomainResolver, CrossDomainEvidence
from mayon_cortex.reasoning.abstraction import FractalAbstractionEngine, AbstractConcept
from mayon_cortex.reasoning.knowledge_correlation import KnowledgeCorrelator, CorrelationPath, SynthesisResult
from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCObject,
    ARCPerception,
    ARCDSL,
    ARCProgramSynthesizer,
    IntelligentSynthesizer,
    ARCTask,
    ARCSolution,
)
from mayon_cortex.reasoning.cortical_arc_reasoner import CorticalARCReasoner, CorticalHypothesis
from mayon_cortex.reasoning.symbolic_algebra import (
    Expr,
    Const,
    Var,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Polynomial,
    UniversalEquationSolver,
    ExprParser,
    to_expr,
)
from mayon_cortex.reasoning.number_theory import NumberTheoryEngine
from mayon_cortex.reasoning.matrix_engine import MatrixEngine, RowOperation
from mayon_cortex.reasoning.calculus_engine import CalculusEngine
from mayon_cortex.reasoning.puzzle_solver import (
    SudokuSolver,
    CryptarithmSolver,
    NQueensSolver,
    MagicSquareEngine,
)

__all__ = [
    "PathRanker",
    "ScoredPath",
    "LogicEngine",
    "ProofStep",
    "FormalProof",
    "ThinkingEngine",
    "ThoughtResult",
    "ProblemSolver",
    "ProblemDecomposition",
    "SolutionStep",
    "SolvedProblem",
    "JudgmentEngine",
    "JudgmentVerdict",
    "AnalogyEngine",
    "AnalogyResult",
    "CrossDomainResolver",
    "CrossDomainEvidence",
    "FractalAbstractionEngine",
    "AbstractConcept",
    "KnowledgeCorrelator",
    "CorrelationPath",
    "SynthesisResult",
    "ARCGrid",
    "ARCObject",
    "ARCPerception",
    "ARCDSL",
    "ARCProgramSynthesizer",
    "IntelligentSynthesizer",
    "CorticalARCReasoner",
    "CorticalHypothesis",
    "ARCTask",
    "ARCSolution",
    "Expr",
    "Const",
    "Var",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Neg",
    "Polynomial",
    "UniversalEquationSolver",
    "ExprParser",
    "to_expr",
    "NumberTheoryEngine",
    "MatrixEngine",
    "RowOperation",
    "CalculusEngine",
    "SudokuSolver",
    "CryptarithmSolver",
    "NQueensSolver",
    "MagicSquareEngine",
]
