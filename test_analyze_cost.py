"""
Test suite for Phase 3B: Cost Analysis

Following TDD methodology - tests written FIRST, then implementation.
Tests for cost analysis functionality including token extraction,
cost calculation, and scaling projections.

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-09
"""

import pytest
from datetime import datetime, timezone
from analyze_cost import (
    PricingConfig,
    TokenUsage,
    CostBreakdown,
    extract_token_usage,
    calculate_trace_cost,
)
from analyze_traces import Trace


class TestPricingConfig:
    """Test PricingConfig dataclass validation."""

    def test_pricing_config_creation_valid(self):
        """Test creating valid pricing config."""
        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.001,
            output_tokens_per_1k=0.002,
            cache_read_per_1k=0.0001,
        )
        assert pricing.model_name == "Test Model"
        assert pricing.input_tokens_per_1k == 0.001
        assert pricing.output_tokens_per_1k == 0.002
        assert pricing.cache_read_per_1k == 0.0001

    def test_pricing_config_without_cache(self):
        """Test pricing config without cache pricing."""
        pricing = PricingConfig(
            model_name="No Cache Model",
            input_tokens_per_1k=0.001,
            output_tokens_per_1k=0.002,
        )
        assert pricing.cache_read_per_1k is None

    def test_pricing_config_negative_input_price_raises(self):
        """Test that negative input price raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PricingConfig(
                model_name="Bad Model",
                input_tokens_per_1k=-0.001,
                output_tokens_per_1k=0.002,
            )

    def test_pricing_config_negative_output_price_raises(self):
        """Test that negative output price raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PricingConfig(
                model_name="Bad Model",
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=-0.002,
            )

    def test_pricing_config_negative_cache_price_raises(self):
        """Test that negative cache price raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PricingConfig(
                model_name="Bad Model",
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=0.002,
                cache_read_per_1k=-0.0001,
            )


class TestTokenExtraction:
    """Test token usage extraction from traces."""

    def test_extract_token_usage_from_outputs(self):
        """Test extracting tokens from trace.outputs['usage_metadata']."""
        trace = Trace(
            id="test-1",
            name="ChatGoogleGenerativeAI",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={
                "usage_metadata": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                }
            },
            error=None,
        )

        result = extract_token_usage(trace)

        assert result is not None
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500
        assert result.cached_tokens is None

    def test_extract_token_usage_with_cache_data(self):
        """Test extracting tokens including cache_read data."""
        trace = Trace(
            id="test-2",
            name="ChatGoogleGenerativeAI",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={
                "usage_metadata": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                    "input_token_details": {
                        "cache_read": 800,
                    },
                }
            },
            error=None,
        )

        result = extract_token_usage(trace)

        assert result is not None
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500
        assert result.cached_tokens == 800

    def test_extract_token_usage_missing_data_returns_none(self):
        """Test that missing token data returns None."""
        trace = Trace(
            id="test-3",
            name="non_llm_node",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        result = extract_token_usage(trace)

        assert result is None

    def test_extract_token_usage_from_inputs_fallback(self):
        """Test fallback to extracting from inputs if not in outputs."""
        trace = Trace(
            id="test-4",
            name="ChatGoogleGenerativeAI",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id=None,
            child_ids=[],
            inputs={
                "usage_metadata": {
                    "input_tokens": 2000,
                    "output_tokens": 1000,
                    "total_tokens": 3000,
                }
            },
            outputs={},
            error=None,
        )

        result = extract_token_usage(trace)

        assert result is not None
        assert result.input_tokens == 2000
        assert result.output_tokens == 1000


class TestCostCalculation:
    """Test cost calculation functions."""

    def test_calculate_trace_cost_basic(self):
        """Test basic cost calculation without cache."""
        token_usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cached_tokens=None,
        )
        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.00125,  # $1.25 per 1M
            output_tokens_per_1k=0.005,   # $5.00 per 1M
        )

        result = calculate_trace_cost(token_usage, pricing)

        # Expected: (1000 * 0.00125 / 1000) + (500 * 0.005 / 1000)
        # = 0.00125 + 0.0025 = 0.00375
        assert result.trace_id is not None
        assert result.input_cost == pytest.approx(0.00125, abs=0.00001)
        assert result.output_cost == pytest.approx(0.0025, abs=0.00001)
        assert result.cache_cost == 0.0
        assert result.total_cost == pytest.approx(0.00375, abs=0.00001)

    def test_calculate_trace_cost_with_cache(self):
        """Test cost calculation with cache reads."""
        token_usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cached_tokens=800,
        )
        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.00125,
            output_tokens_per_1k=0.005,
            cache_read_per_1k=0.0003125,  # $0.3125 per 1M
        )

        result = calculate_trace_cost(token_usage, pricing)

        # Expected cache cost: 800 * 0.0003125 / 1000 = 0.00025
        assert result.cache_cost == pytest.approx(0.00025, abs=0.00001)
        # Total: 0.00125 + 0.0025 + 0.00025 = 0.00400
        assert result.total_cost == pytest.approx(0.00400, abs=0.00001)

    def test_calculate_trace_cost_zero_tokens(self):
        """Test handling of zero token usage."""
        token_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cached_tokens=None,
        )
        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.00125,
            output_tokens_per_1k=0.005,
        )

        result = calculate_trace_cost(token_usage, pricing)

        assert result.input_cost == 0.0
        assert result.output_cost == 0.0
        assert result.total_cost == 0.0


class TestWorkflowCostAnalysis:
    """Test workflow-level cost analysis."""

    def test_calculate_workflow_cost_single_trace(self):
        """Test calculating cost for workflow with one LLM trace."""
        from analyze_traces import Workflow

        # Create root trace (no cost)
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["child-1"],
            inputs={},
            outputs={},
            error=None,
        )

        # Create child LLM trace with cost
        child = Trace(
            id="child-1",
            name="ChatGoogleGenerativeAI",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={
                "usage_metadata": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                }
            },
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={"ChatGoogleGenerativeAI": [child]},
            all_traces=[root, child],
        )

        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.00125,
            output_tokens_per_1k=0.005,
        )

        from analyze_cost import calculate_workflow_cost

        result = calculate_workflow_cost(workflow, pricing)

        assert result is not None
        assert result.workflow_id == "root-1"
        assert len(result.node_costs) == 1
        assert result.total_cost == pytest.approx(0.00375, abs=0.00001)

    def test_calculate_workflow_cost_no_token_data(self):
        """Test workflow with no token data returns None."""
        from analyze_traces import Workflow

        root = Trace(
            id="root-2",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={},
            all_traces=[root],
        )

        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.00125,
            output_tokens_per_1k=0.005,
        )

        from analyze_cost import calculate_workflow_cost

        result = calculate_workflow_cost(workflow, pricing)

        # Should return result with zero cost, not None
        assert result is not None
        assert result.total_cost == 0.0
        assert len(result.node_costs) == 0


class TestScalingProjections:
    """Test scaling cost projections."""

    def test_project_scaling_costs_basic(self):
        """Test basic scaling projections at 10x, 100x, 1000x."""
        from analyze_cost import project_scaling_costs, SCALING_FACTORS

        avg_cost_per_workflow = 0.01  # $0.01 per workflow
        current_workflow_count = 100

        result = project_scaling_costs(
            avg_cost_per_workflow=avg_cost_per_workflow,
            current_workflow_count=current_workflow_count,
            scaling_factors=SCALING_FACTORS,
        )

        assert result is not None
        assert "1x" in result
        assert "10x" in result
        assert "100x" in result
        assert "1000x" in result

        # Verify 1x (current)
        assert result["1x"].scale_factor == 1
        assert result["1x"].workflow_count == 100
        assert result["1x"].total_cost == pytest.approx(1.0, abs=0.01)  # 100 * $0.01

        # Verify 10x
        assert result["10x"].scale_factor == 10
        assert result["10x"].workflow_count == 1000
        assert result["10x"].total_cost == pytest.approx(10.0, abs=0.01)

        # Verify 100x
        assert result["100x"].scale_factor == 100
        assert result["100x"].workflow_count == 10000
        assert result["100x"].total_cost == pytest.approx(100.0, abs=0.01)

        # Verify 1000x
        assert result["1000x"].scale_factor == 1000
        assert result["1000x"].workflow_count == 100000
        assert result["1000x"].total_cost == pytest.approx(1000.0, abs=0.01)

    def test_project_scaling_costs_with_monthly(self):
        """Test that monthly costs are calculated correctly."""
        from analyze_cost import project_scaling_costs

        avg_cost_per_workflow = 0.05
        current_workflow_count = 50
        monthly_estimate = 500  # Assume 500 workflows per month

        result = project_scaling_costs(
            avg_cost_per_workflow=avg_cost_per_workflow,
            current_workflow_count=current_workflow_count,
            scaling_factors=[1, 10],
            monthly_workflow_estimate=monthly_estimate,
        )

        # At 1x: 500 workflows/month * $0.05 = $25/month
        assert result["1x"].cost_per_month_30days is not None
        assert result["1x"].cost_per_month_30days == pytest.approx(25.0, abs=0.01)

        # At 10x: 5000 workflows/month * $0.05 = $250/month
        assert result["10x"].cost_per_month_30days is not None
        assert result["10x"].cost_per_month_30days == pytest.approx(250.0, abs=0.01)

    def test_project_scaling_costs_zero_cost(self):
        """Test handling of zero cost per workflow."""
        from analyze_cost import project_scaling_costs

        result = project_scaling_costs(
            avg_cost_per_workflow=0.0,
            current_workflow_count=100,
            scaling_factors=[1, 10],
        )

        assert result["1x"].total_cost == 0.0
        assert result["10x"].total_cost == 0.0


class TestNodeCostAggregation:
    """Test node-level cost aggregation across workflows."""

    def test_aggregate_node_costs_multiple_workflows(self):
        """Test aggregating costs by node type across multiple workflows."""
        from analyze_cost import aggregate_node_costs, WorkflowCostAnalysis, CostBreakdown, TokenUsage

        # Create mock workflow analyses
        workflow1_costs = [
            CostBreakdown(
                trace_id="1",
                trace_name="ChatModel",
                input_cost=0.001,
                output_cost=0.002,
                cache_cost=0.0,
                total_cost=0.003,
                token_usage=TokenUsage(1000, 500, 1500),
            ),
            CostBreakdown(
                trace_id="2",
                trace_name="Validator",
                input_cost=0.0005,
                output_cost=0.001,
                cache_cost=0.0,
                total_cost=0.0015,
                token_usage=TokenUsage(500, 250, 750),
            ),
        ]

        workflow2_costs = [
            CostBreakdown(
                trace_id="3",
                trace_name="ChatModel",
                input_cost=0.002,
                output_cost=0.003,
                cache_cost=0.0,
                total_cost=0.005,
                token_usage=TokenUsage(2000, 1000, 3000),
            ),
        ]

        workflows = [
            WorkflowCostAnalysis("wf1", 0.0045, workflow1_costs, 2250),
            WorkflowCostAnalysis("wf2", 0.005, workflow2_costs, 3000),
        ]

        result = aggregate_node_costs(workflows)

        assert result is not None
        assert len(result) == 2  # ChatModel and Validator

        # Should be sorted by total cost descending
        assert result[0].node_name == "ChatModel"
        assert result[0].execution_count == 2
        assert result[0].total_cost == pytest.approx(0.008, abs=0.0001)
        assert result[0].avg_cost_per_execution == pytest.approx(0.004, abs=0.0001)

        assert result[1].node_name == "Validator"
        assert result[1].execution_count == 1
        assert result[1].total_cost == pytest.approx(0.0015, abs=0.0001)

    def test_aggregate_node_costs_empty_workflows(self):
        """Test aggregating with no workflows."""
        from analyze_cost import aggregate_node_costs

        result = aggregate_node_costs([])

        assert result is not None
        assert len(result) == 0


class TestMainAnalysisFunction:
    """Test main analyze_costs() orchestration function."""

    def test_analyze_costs_integration(self):
        """Test complete cost analysis workflow."""
        from analyze_cost import analyze_costs, PricingConfig
        from analyze_traces import Workflow, Trace
        from datetime import datetime, timezone

        # Create minimal workflow for testing
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["child-1"],
            inputs={},
            outputs={},
            error=None,
        )

        child = Trace(
            id="child-1",
            name="ChatModel",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={
                "usage_metadata": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                }
            },
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={"ChatModel": [child]},
            all_traces=[root, child],
        )

        pricing = PricingConfig(
            model_name="Test Model",
            input_tokens_per_1k=0.001,
            output_tokens_per_1k=0.002,
        )

        result = analyze_costs([workflow], pricing)

        assert result is not None
        assert result.total_workflows_analyzed == 1
        assert result.avg_cost_per_workflow > 0
        assert len(result.node_summaries) == 1
        assert result.node_summaries[0].node_name == "ChatModel"
        assert len(result.scaling_projections) > 0
        assert "1x" in result.scaling_projections


# Run tests with: pytest test_analyze_cost.py -v
