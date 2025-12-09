"""
LangSmith Trace Cost Analysis Tool - Phase 3B

This module provides cost analysis capabilities for LangSmith trace exports.
Calculates costs based on token usage with configurable pricing models.

Following PDCA (Plan-Do-Check-Act) methodology with TDD approach.

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-09
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from analyze_traces import Trace, Workflow


# ============================================================================
# Pricing Configuration
# ============================================================================

@dataclass
class PricingConfig:
    """Configurable pricing model for any LLM provider."""

    model_name: str
    input_tokens_per_1k: float           # Cost per 1K input tokens
    output_tokens_per_1k: float          # Cost per 1K output tokens
    cache_read_per_1k: Optional[float] = None   # Cost per 1K cache read tokens (if applicable)

    def __post_init__(self):
        """Validate pricing configuration."""
        if self.input_tokens_per_1k < 0 or self.output_tokens_per_1k < 0:
            raise ValueError("Token prices must be non-negative")
        if self.cache_read_per_1k is not None and self.cache_read_per_1k < 0:
            raise ValueError("Cache read price must be non-negative")


# Example pricing configs for reference (NOT hard-coded defaults)
# Users should create their own PricingConfig instances
EXAMPLE_PRICING_CONFIGS = {
    "gemini_1.5_pro": PricingConfig(
        model_name="Gemini 1.5 Pro",
        input_tokens_per_1k=0.00125,      # $1.25 per 1M input tokens
        output_tokens_per_1k=0.005,       # $5.00 per 1M output tokens
        cache_read_per_1k=0.0003125,      # $0.3125 per 1M cache read tokens
    ),
}

SCALING_FACTORS = [1, 10, 100, 1000]  # Current, 10x, 100x, 1000x


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage for a single trace."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None      # From input_token_details.cache_read

    def has_cache_data(self) -> bool:
        """Check if cache token data is available."""
        return self.cached_tokens is not None


@dataclass
class CostBreakdown:
    """Cost breakdown for a single trace."""

    trace_id: str
    trace_name: str
    input_cost: float
    output_cost: float
    cache_cost: float
    total_cost: float
    token_usage: TokenUsage

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for CSV/reporting."""
        return {
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "input_tokens": self.token_usage.input_tokens,
            "output_tokens": self.token_usage.output_tokens,
            "total_tokens": self.token_usage.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_cost": self.cache_cost,
            "total_cost": self.total_cost,
        }


@dataclass
class WorkflowCostAnalysis:
    """Cost analysis for a single workflow."""

    workflow_id: str
    total_cost: float
    node_costs: List[CostBreakdown]
    total_tokens: int
    cache_effectiveness_percent: Optional[float] = None  # If cache data available


@dataclass
class ScalingProjection:
    """Cost projection at a specific scale factor."""

    scale_factor: int
    workflow_count: int
    total_cost: float
    cost_per_month_30days: Optional[float] = None  # If monthly estimate provided


@dataclass
class NodeCostSummary:
    """Cost summary for a node type across workflows."""

    node_name: str
    execution_count: int
    total_cost: float
    avg_cost_per_execution: float
    percent_of_total_cost: float


@dataclass
class CostAnalysisResults:
    """Complete cost analysis results."""

    # Per-workflow metrics
    avg_cost_per_workflow: float
    median_cost_per_workflow: float
    min_cost: float
    max_cost: float

    # Node-level breakdown
    node_summaries: List[NodeCostSummary]    # Sorted by cost descending
    top_cost_driver: Optional[str]

    # Cache effectiveness
    cache_effectiveness_percent: Optional[float]
    cache_savings_dollars: Optional[float]

    # Scaling projections
    scaling_projections: Dict[str, ScalingProjection]  # "10x", "100x", etc.

    # Metadata
    total_workflows_analyzed: int
    data_quality_notes: List[str]


# ============================================================================
# Token Extraction Functions
# ============================================================================

def extract_token_usage(trace: Trace) -> Optional[TokenUsage]:
    """
    Extract token usage from trace outputs/inputs.

    Checks multiple possible locations:
    1. trace.outputs["usage_metadata"]
    2. trace.inputs["usage_metadata"] (fallback)
    3. Extracts cache_read from input_token_details if available

    Args:
        trace: Trace object to extract from

    Returns:
        TokenUsage if data found, None otherwise
    """
    # Try outputs first
    usage_data = trace.outputs.get("usage_metadata")

    # Fallback to inputs
    if not usage_data:
        usage_data = trace.inputs.get("usage_metadata")

    if not usage_data:
        return None

    # Safely extract with defaults
    input_tokens = usage_data.get("input_tokens", 0)
    output_tokens = usage_data.get("output_tokens", 0)
    total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)

    # Extract cache tokens if available
    cached_tokens = None
    if "input_token_details" in usage_data:
        token_details = usage_data["input_token_details"]
        if isinstance(token_details, dict):
            cached_tokens = token_details.get("cache_read")

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )


# ============================================================================
# Cost Calculation Functions
# ============================================================================

def calculate_trace_cost(
    token_usage: TokenUsage,
    pricing_config: PricingConfig,
    trace_id: str = "",
    trace_name: str = "",
) -> CostBreakdown:
    """
    Calculate cost for single trace using pricing model.

    Args:
        token_usage: Token usage data
        pricing_config: Pricing configuration
        trace_id: Optional trace ID for breakdown
        trace_name: Optional trace name for breakdown

    Returns:
        CostBreakdown with detailed cost information
    """
    # Calculate input cost: (tokens / 1000) * price_per_1k
    input_cost = (token_usage.input_tokens / 1000.0) * pricing_config.input_tokens_per_1k

    # Calculate output cost
    output_cost = (token_usage.output_tokens / 1000.0) * pricing_config.output_tokens_per_1k

    # Calculate cache cost if applicable
    cache_cost = 0.0
    if (token_usage.cached_tokens is not None and
        pricing_config.cache_read_per_1k is not None):
        cache_cost = (token_usage.cached_tokens / 1000.0) * pricing_config.cache_read_per_1k

    total_cost = input_cost + output_cost + cache_cost

    return CostBreakdown(
        trace_id=trace_id,
        trace_name=trace_name,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_cost=cache_cost,
        total_cost=total_cost,
        token_usage=token_usage,
    )


def calculate_workflow_cost(
    workflow: Workflow,
    pricing_config: PricingConfig,
) -> WorkflowCostAnalysis:
    """
    Calculate total cost and breakdown by node for a workflow.

    Args:
        workflow: Workflow to analyze
        pricing_config: Pricing configuration

    Returns:
        WorkflowCostAnalysis with cost breakdown
    """
    node_costs = []
    total_cost = 0.0
    total_tokens = 0

    # Process all traces in workflow
    for trace in workflow.all_traces:
        token_usage = extract_token_usage(trace)
        if token_usage:
            cost_breakdown = calculate_trace_cost(
                token_usage,
                pricing_config,
                trace_id=trace.id,
                trace_name=trace.name,
            )
            node_costs.append(cost_breakdown)
            total_cost += cost_breakdown.total_cost
            total_tokens += token_usage.total_tokens

    return WorkflowCostAnalysis(
        workflow_id=workflow.root_trace.id,
        total_cost=total_cost,
        node_costs=node_costs,
        total_tokens=total_tokens,
    )


# ============================================================================
# Scaling Projection Functions
# ============================================================================

def project_scaling_costs(
    avg_cost_per_workflow: float,
    current_workflow_count: int,
    scaling_factors: List[int],
    monthly_workflow_estimate: Optional[int] = None,
) -> Dict[str, ScalingProjection]:
    """
    Project costs at different scale factors (1x, 10x, 100x, 1000x).

    Args:
        avg_cost_per_workflow: Average cost per workflow in dollars
        current_workflow_count: Current number of workflows in dataset
        scaling_factors: List of scale factors (e.g., [1, 10, 100, 1000])
        monthly_workflow_estimate: Optional monthly workflow estimate for monthly cost

    Returns:
        Dict mapping scale labels ("1x", "10x", etc.) to ScalingProjection objects
    """
    projections = {}

    for factor in scaling_factors:
        scaled_workflow_count = current_workflow_count * factor
        total_cost = avg_cost_per_workflow * scaled_workflow_count

        # Calculate monthly cost if estimate provided
        cost_per_month = None
        if monthly_workflow_estimate is not None:
            monthly_workflows_at_scale = monthly_workflow_estimate * factor
            cost_per_month = avg_cost_per_workflow * monthly_workflows_at_scale

        projection = ScalingProjection(
            scale_factor=factor,
            workflow_count=scaled_workflow_count,
            total_cost=total_cost,
            cost_per_month_30days=cost_per_month,
        )

        # Create label (1x, 10x, 100x, etc.)
        label = f"{factor}x"
        projections[label] = projection

    return projections


# ============================================================================
# Aggregation and Analysis Functions
# ============================================================================

def aggregate_node_costs(
    workflow_analyses: List[WorkflowCostAnalysis],
) -> List[NodeCostSummary]:
    """
    Aggregate costs by node type across all workflows.

    Args:
        workflow_analyses: List of WorkflowCostAnalysis objects

    Returns:
        List of NodeCostSummary objects sorted by total_cost descending
    """
    if not workflow_analyses:
        return []

    # Aggregate by node name
    node_data: Dict[str, Dict[str, Any]] = {}
    total_overall_cost = 0.0

    for workflow_analysis in workflow_analyses:
        for cost_breakdown in workflow_analysis.node_costs:
            node_name = cost_breakdown.trace_name
            if node_name not in node_data:
                node_data[node_name] = {
                    "execution_count": 0,
                    "total_cost": 0.0,
                }
            node_data[node_name]["execution_count"] += 1
            node_data[node_name]["total_cost"] += cost_breakdown.total_cost
            total_overall_cost += cost_breakdown.total_cost

    # Create summaries with percentages
    summaries = []
    for node_name, data in node_data.items():
        execution_count = data["execution_count"]
        total_cost = data["total_cost"]
        avg_cost = total_cost / execution_count if execution_count > 0 else 0.0
        percent_of_total = (total_cost / total_overall_cost * 100.0) if total_overall_cost > 0 else 0.0

        summary = NodeCostSummary(
            node_name=node_name,
            execution_count=execution_count,
            total_cost=total_cost,
            avg_cost_per_execution=avg_cost,
            percent_of_total_cost=percent_of_total,
        )
        summaries.append(summary)

    # Sort by total cost descending
    summaries.sort(key=lambda s: s.total_cost, reverse=True)

    return summaries


def analyze_costs(
    workflows: List[Workflow],
    pricing_config: PricingConfig,
    scaling_factors: Optional[List[int]] = None,
    monthly_workflow_estimate: Optional[int] = None,
) -> CostAnalysisResults:
    """
    Perform complete cost analysis on workflows.
    Main entry point for Phase 3B.

    Args:
        workflows: List of Workflow objects to analyze
        pricing_config: Pricing configuration for cost calculation
        scaling_factors: Optional list of scale factors (default: [1, 10, 100, 1000])
        monthly_workflow_estimate: Optional monthly workflow estimate for projections

    Returns:
        CostAnalysisResults with complete analysis
    """
    if scaling_factors is None:
        scaling_factors = SCALING_FACTORS

    data_quality_notes = []

    # Calculate cost for each workflow
    workflow_analyses = []
    workflow_costs = []

    for workflow in workflows:
        analysis = calculate_workflow_cost(workflow, pricing_config)
        workflow_analyses.append(analysis)
        workflow_costs.append(analysis.total_cost)

    # Calculate per-workflow statistics
    if workflow_costs:
        avg_cost = sum(workflow_costs) / len(workflow_costs)
        sorted_costs = sorted(workflow_costs)
        median_cost = sorted_costs[len(sorted_costs) // 2]
        min_cost = min(workflow_costs)
        max_cost = max(workflow_costs)
    else:
        avg_cost = median_cost = min_cost = max_cost = 0.0
        data_quality_notes.append("No workflows with cost data found")

    # Aggregate node costs
    node_summaries = aggregate_node_costs(workflow_analyses)
    top_cost_driver = node_summaries[0].node_name if node_summaries else None

    # Calculate scaling projections
    scaling_projections = project_scaling_costs(
        avg_cost_per_workflow=avg_cost,
        current_workflow_count=len(workflows),
        scaling_factors=scaling_factors,
        monthly_workflow_estimate=monthly_workflow_estimate,
    )

    # Cache effectiveness (not yet implemented - Phase 3B extension)
    cache_effectiveness_percent = None
    cache_savings_dollars = None

    return CostAnalysisResults(
        avg_cost_per_workflow=avg_cost,
        median_cost_per_workflow=median_cost,
        min_cost=min_cost,
        max_cost=max_cost,
        node_summaries=node_summaries,
        top_cost_driver=top_cost_driver,
        cache_effectiveness_percent=cache_effectiveness_percent,
        cache_savings_dollars=cache_savings_dollars,
        scaling_projections=scaling_projections,
        total_workflows_analyzed=len(workflows),
        data_quality_notes=data_quality_notes,
    )
