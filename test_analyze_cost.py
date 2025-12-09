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


# Run tests with: pytest test_analyze_cost.py -v
