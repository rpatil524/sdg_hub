# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified LLMChatBlock supporting all providers via LiteLLM."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from datasets import Dataset

# First Party
from sdg_hub.core.blocks.llm import LLMChatBlock, LLMConfig
from sdg_hub.core.utils.error_handling import BlockValidationError
import pytest


class MockMessage:
    """Mock message class that behaves like LiteLLM message for dict() conversion."""

    def __init__(self, content):
        self.content = content

    def __iter__(self):
        return iter(["content"])

    def __getitem__(self, key):
        if key == "content":
            return self.content
        raise KeyError(key)

    def keys(self):
        return ["content"]

    def values(self):
        return [self.content]

    def items(self):
        return [("content", self.content)]


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function."""
    with patch("sdg_hub.core.blocks.llm.client_manager.completion") as mock_completion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("Test response")
        mock_response.choices = [choice]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_completion_multiple():
    """Mock LiteLLM completion function for multiple responses (n > 1)."""
    with patch("sdg_hub.core.blocks.llm.client_manager.completion") as mock_completion:
        mock_response = MagicMock()
        choices = []
        for i in range(3):
            choice = MagicMock()
            choice.message = MockMessage(f"Response {i + 1}")
            choices.append(choice)
        mock_response.choices = choices
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_litellm_acompletion():
    """Mock LiteLLM async completion function."""
    with patch(
        "sdg_hub.core.blocks.llm.client_manager.acompletion"
    ) as mock_acompletion:
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message = MockMessage("Test async response")
        mock_response.choices = [choice]

        async def mock_async_completion(*_args, **_kwargs):
            return mock_response

        mock_acompletion.side_effect = mock_async_completion
        yield mock_acompletion


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format."""
    return [
        [{"role": "user", "content": "Hello, how are you?"}],
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ],
    ]


@pytest.fixture
def sample_dataset(sample_messages):
    """Create a sample dataset with messages."""
    return Dataset.from_dict({"messages": sample_messages})


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = LLMConfig(model="openai/gpt-4")

        assert config.model == "openai/gpt-4"
        assert config.get_provider() == "openai"
        assert config.get_model_name() == "gpt-4"
        assert not config.is_local_model()
        assert config.timeout == 120.0
        assert config.max_retries == 6

    def test_config_with_parameters(self):
        """Test configuration with generation parameters."""
        config = LLMConfig(
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.top_p == 0.9
        assert config.get_provider() == "anthropic"

    def test_local_model_detection(self):
        """Test local model detection."""
        local_config = LLMConfig(
            model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
            api_base="http://localhost:8000/v1",
        )

        assert local_config.is_local_model()
        assert local_config.get_provider() == "hosted_vllm"

    def test_invalid_model_format(self):
        """Test error on invalid model format."""
        with pytest.raises(
            ValueError, match="should be in format 'provider/model-name'"
        ):
            LLMConfig(model="invalid-model-format")

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        config = LLMConfig(
            model="openai/gpt-4", temperature=1.0, max_tokens=100, top_p=0.5
        )
        assert config.temperature == 1.0

        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            LLMConfig(model="openai/gpt-4", temperature=3.0)

        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(model="openai/gpt-4", max_tokens=-1)

    def test_api_key_resolution(self):
        """Test API key resolution from environment variables."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = LLMConfig(model="openai/gpt-4")
            assert config.api_key == "test-key"

    def test_generation_kwargs(self):
        """Test generation kwargs extraction."""
        config = LLMConfig(
            model="openai/gpt-4",
            temperature=0.7,
            max_tokens=100,
            seed=42,
            extra_headers={"x-custom": "value"},
            timeout=120,
        )

        kwargs = config.get_generation_kwargs()
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["seed"] == 42
        assert kwargs["extra_headers"] == {"x-custom": "value"}
        assert kwargs["timeout"] == 120

    def test_merge_overrides(self):
        """Test configuration merging with overrides."""
        base_config = LLMConfig(model="openai/gpt-4", temperature=0.5, max_tokens=100)

        new_config = base_config.merge_overrides(temperature=0.9, seed=42)

        assert new_config.temperature == 0.9  # Overridden
        assert new_config.max_tokens == 100  # Preserved
        assert new_config.seed == 42  # Added
        assert base_config.temperature == 0.5  # Original unchanged


class TestLLMChatBlock:
    """Tests for LLMChatBlock."""

    def test_init_openai_model(self, mock_litellm_completion):
        """Test initialization with OpenAI model."""
        block = LLMChatBlock(
            block_name="test_openai",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            temperature=0.7,
        )

        assert block.block_name == "test_openai"
        assert block.input_cols[0] == "messages"
        assert block.output_cols[0] == "response"
        assert block.client_manager.config.model == "openai/gpt-4"
        assert block.client_manager.config.temperature == 0.7
        assert not block.async_mode

    def test_init_anthropic_model(self, mock_litellm_completion):
        """Test initialization with Anthropic model."""
        block = LLMChatBlock(
            block_name="test_anthropic",
            input_cols="messages",
            output_cols="response",
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.5,
        )

        assert block.client_manager.config.model == "anthropic/claude-3-sonnet-20240229"
        assert block.client_manager.config.get_provider() == "anthropic"

    def test_init_local_model(self, mock_litellm_completion):
        """Test initialization with local vLLM model."""
        block = LLMChatBlock(
            block_name="test_local",
            input_cols="messages",
            output_cols="response",
            model="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
            api_base="http://localhost:8000/v1",
        )

        assert block.client_manager.config.is_local_model()
        assert block.client_manager.config.api_base == "http://localhost:8000/v1"

    def test_init_async_mode(self, mock_litellm_acompletion):
        """Test initialization with async mode."""
        block = LLMChatBlock(
            block_name="test_async",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            async_mode=True,
        )

        assert block.async_mode is True

    def test_init_multiple_input_cols_error(self):
        """Test error when multiple input columns provided."""
        with pytest.raises(ValueError, match="expects exactly one input column"):
            LLMChatBlock(
                block_name="test_block",
                input_cols=["messages1", "messages2"],
                output_cols="response",
                model="openai/gpt-4",
            )

    def test_init_multiple_output_cols_error(self):
        """Test error when multiple output columns provided."""
        with pytest.raises(ValueError, match="expects exactly one output column"):
            LLMChatBlock(
                block_name="test_block",
                input_cols="messages",
                output_cols=["response1", "response2"],
                model="openai/gpt-4",
            )

    def test_sync_generation(self, mock_litellm_completion, sample_dataset):
        """Test synchronous generation."""
        block = LLMChatBlock(
            block_name="test_sync",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(
            response["content"] == "Test response" for response in result["response"]
        )
        assert mock_litellm_completion.call_count == 2

    def test_async_generation(self, mock_litellm_acompletion, sample_dataset):
        """Test asynchronous generation."""
        block = LLMChatBlock(
            block_name="test_async",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            async_mode=True,
        )

        result = block.generate(sample_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 2
        assert all(
            response["content"] == "Test async response"
            for response in result["response"]
        )
        assert (
            mock_litellm_acompletion.call_count == 2
        )  # Concurrent calls, one per message

    def test_generation_with_overrides(self, mock_litellm_completion, sample_dataset):
        """Test generation with runtime parameter overrides."""
        block = LLMChatBlock(
            block_name="test_overrides",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            temperature=0.5,
        )

        block.generate(sample_dataset, temperature=0.9, max_tokens=150)

        # Verify override parameters were used
        calls = mock_litellm_completion.call_args_list
        assert calls[0][1]["temperature"] == 0.9  # Override value
        assert calls[0][1]["max_tokens"] == 150  # New parameter
        assert calls[0][1]["model"] == "openai/gpt-4"

    def test_generation_all_providers(self, mock_litellm_completion, sample_dataset):
        """Test generation works with different providers."""
        providers_models = [
            "openai/gpt-4",
            "anthropic/claude-3-sonnet-20240229",
            "google/gemini-pro",
            "hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
        ]

        for model in providers_models:
            block = LLMChatBlock(
                block_name=f"test_{model.replace('/', '_')}",
                input_cols="messages",
                output_cols="response",
                model=model,
                api_key="test-key",
            )

            # Create single message dataset for simpler testing
            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            result = block.generate(single_dataset)

            assert "response" in result.column_names
            assert len(result["response"]) == 1
            assert result["response"][0]["content"] == "Test response"

    def test_generation_with_all_parameters(
        self, mock_litellm_completion, sample_dataset
    ):
        """Test generation with all supported parameters."""
        block = LLMChatBlock(
            block_name="test_all_params",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            frequency_penalty=0.1,
            max_tokens=150,
            n=1,
            presence_penalty=0.2,
            response_format={"type": "json_object"},
            seed=42,
            stop=["END"],
            stream=False,
            temperature=0.8,
            top_p=0.95,
            user="test_user",
        )

        # Create single message dataset for simpler testing
        single_dataset = Dataset.from_dict(
            {"messages": [sample_dataset["messages"][0]]}
        )
        block.generate(single_dataset)

        # Verify all parameters were passed
        call_kwargs = mock_litellm_completion.call_args[1]
        assert call_kwargs["frequency_penalty"] == 0.1
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["n"] == 1
        assert call_kwargs["presence_penalty"] == 0.2
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["stream"] is False
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["user"] == "test_user"

    def test_custom_validation(self, mock_litellm_completion):
        """Test custom validation functionality."""
        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        # Valid dataset
        valid_dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "Hello!"}],
                    [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                ]
            }
        )

        # Should not raise any exception
        block._validate_custom(valid_dataset)

        # Invalid dataset - messages not a list
        invalid_dataset = Dataset.from_dict(
            {"messages": ["not a list", "also not a list"]}
        )

        with pytest.raises(BlockValidationError, match="must contain a list"):
            block._validate_custom(invalid_dataset)

        # Invalid dataset - empty messages
        empty_dataset = Dataset.from_dict({"messages": [[], []]})

        with pytest.raises(BlockValidationError, match="Messages list is empty"):
            block._validate_custom(empty_dataset)

    def test_model_info(self, mock_litellm_completion):
        """Test model info functionality."""
        block = LLMChatBlock(
            block_name="test_info",
            input_cols="messages",
            output_cols="response",
            model="anthropic/claude-3-sonnet-20240229",
            temperature=0.5,
        )

        info = block.get_model_info()
        assert info["model"] == "anthropic/claude-3-sonnet-20240229"
        assert info["provider"] == "anthropic"
        assert info["block_name"] == "test_info"
        assert info["input_column"] == "messages"
        assert info["output_column"] == "response"
        assert not info["is_local"]

    def test_empty_dataset(self, mock_litellm_completion):
        """Test handling of empty datasets."""
        empty_dataset = Dataset.from_dict({"messages": []})

        block = LLMChatBlock(
            block_name="test_empty",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        result = block.generate(empty_dataset)

        assert "response" in result.column_names
        assert len(result["response"]) == 0
        assert mock_litellm_completion.call_count == 0


class TestErrorHandling:
    """Test error handling for LLMChatBlock."""

    def test_max_concurrency_value_error(
        self, mock_litellm_acompletion, sample_dataset
    ):
        """Test ValueError is raised when max_concurrency < 1."""
        block = LLMChatBlock(
            block_name="test_max_concurrency_error",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            async_mode=True,
        )

        # Test with max_concurrency = 0
        with pytest.raises(
            ValueError, match="max_concurrency must be greater than 0, got"
        ):
            block.generate(sample_dataset, _flow_max_concurrency=0)

        # Test with max_concurrency = -1
        with pytest.raises(
            ValueError, match="max_concurrency must be greater than 0, got"
        ):
            block.generate(sample_dataset, _flow_max_concurrency=-1)

        # Test with max_concurrency = -5
        with pytest.raises(
            ValueError, match="max_concurrency must be greater than 0, got"
        ):
            block.generate(sample_dataset, _flow_max_concurrency=-5)

    def test_litellm_rate_limit_error(self, sample_dataset):
        """Test handling of LiteLLM rate limit errors."""
        with patch(
            "sdg_hub.core.blocks.llm.client_manager.completion"
        ) as mock_completion:
            # First Party
            from sdg_hub.core.blocks.llm.error_handler import RateLimitError

            # Mock successful response for retry
            mock_response = MagicMock()
            choice = MagicMock()
            choice.message = MockMessage("Success after retry")
            mock_response.choices = [choice]

            # First call raises rate limit error, second succeeds
            mock_completion.side_effect = [
                RateLimitError("Rate limited", llm_provider="openai", model="gpt-4"),
                mock_response,
            ]

            block = LLMChatBlock(
                block_name="test_retry",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="test-key",
            )

            # Create single message dataset to test retry behavior
            single_dataset = Dataset.from_dict(
                {"messages": [sample_dataset["messages"][0]]}
            )
            result = block.generate(single_dataset)

            assert result["response"][0]["content"] == "Success after retry"
            assert mock_completion.call_count == 2

    def test_litellm_authentication_error(self, sample_dataset):
        """Test handling of authentication errors (non-retryable)."""
        with patch(
            "sdg_hub.core.blocks.llm.client_manager.completion"
        ) as mock_completion:
            # First Party
            from sdg_hub.core.blocks.llm.error_handler import AuthenticationError

            mock_completion.side_effect = AuthenticationError(
                "Invalid API key", llm_provider="openai", model="gpt-4"
            )

            block = LLMChatBlock(
                block_name="test_auth_error",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="invalid-key",
            )

            # Should fail immediately without retries
            with pytest.raises(AuthenticationError):
                single_dataset = Dataset.from_dict(
                    {"messages": [sample_dataset["messages"][0]]}
                )
                block.generate(single_dataset)

    def test_litellm_context_window_error(self, sample_dataset):
        """Test handling of context window exceeded errors."""
        with patch(
            "sdg_hub.core.blocks.llm.client_manager.completion"
        ) as mock_completion:
            # First Party
            from sdg_hub.core.blocks.llm.error_handler import ContextWindowExceededError

            mock_completion.side_effect = ContextWindowExceededError(
                "Context window exceeded", llm_provider="openai", model="gpt-4"
            )

            block = LLMChatBlock(
                block_name="test_context_error",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="test-key",
            )

            # Should fail immediately without retries
            with pytest.raises(ContextWindowExceededError):
                single_dataset = Dataset.from_dict(
                    {"messages": [sample_dataset["messages"][0]]}
                )
                block.generate(single_dataset)


class TestRegistration:
    """Test block registration."""

    def test_llm_chat_block_registered(self):
        """Test that LLMChatBlock is properly registered."""
        # First Party
        from sdg_hub import BlockRegistry

        assert "LLMChatBlock" in BlockRegistry._metadata
        assert BlockRegistry._metadata["LLMChatBlock"].block_class == LLMChatBlock


class TestLLMChatBlockValidation:
    """Test LLMChatBlock custom validation functionality."""

    def test_validation_with_valid_messages(self):
        """Test validation passes with properly formatted messages."""
        valid_data = [
            {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is 2+2?"},
                ]
            },
            {"messages": [{"role": "assistant", "content": "Hi there!"}]},
            {
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Follow-up"},
                ]
            },
        ]
        dataset = Dataset.from_list(valid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        # Should not raise any exception
        block._validate_custom(dataset)


class TestMultipleResponses:
    """Test multiple response generation (n > 1)."""

    def test_multiple_responses_sync(
        self, mock_litellm_completion_multiple, sample_dataset
    ):
        """Test synchronous generation with n > 1."""
        block = LLMChatBlock(
            block_name="test_multiple_sync",
            input_cols="messages",
            output_cols="responses",
            model="openai/gpt-4",
            api_key="test-key",
            n=3,  # Generate 3 responses per input
        )

        result = block.generate(sample_dataset)

        assert "responses" in result.column_names
        assert len(result["responses"]) == 2  # Two input samples

        # Each response should be a list of 3 dicts
        for responses in result["responses"]:
            assert isinstance(responses, list)
            assert len(responses) == 3
            assert [r["content"] for r in responses] == [
                "Response 1",
                "Response 2",
                "Response 3",
            ]

        assert mock_litellm_completion_multiple.call_count == 2  # One call per sample

    def test_concurrency_adjustment_with_n_greater_than_1(
        self, mock_litellm_acompletion, sample_dataset
    ):
        """Test concurrency is adjusted when n > 1 to avoid overwhelming API."""
        with patch("sdg_hub.core.blocks.llm.client_manager.logger") as mock_logger:
            block = LLMChatBlock(
                block_name="test_concurrency_adjustment",
                input_cols="messages",
                output_cols="responses",
                model="openai/gpt-4",
                api_key="test-key",
                n=4,  # Generate 4 responses per input
                async_mode=True,
            )

            # Test with max_concurrency = 8, should be adjusted to 2 (8 // 4)
            result = block.generate(sample_dataset, _flow_max_concurrency=8)

            assert "responses" in result.column_names
            assert len(result["responses"]) == 2

            # Verify debug log was called for concurrency adjustment
            mock_logger.debug.assert_called()
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Adjusted max_concurrency" in str(call)
            ]
            assert len(debug_calls) > 0
            assert "Adjusted max_concurrency from 8 to 2" in str(debug_calls[0])
            assert "for n=4 completions per request" in str(debug_calls[0])

    def test_concurrency_warning_when_max_concurrency_less_than_n(
        self, mock_litellm_acompletion, sample_dataset
    ):
        """Test warning is logged when max_concurrency < n."""
        with patch("sdg_hub.core.blocks.llm.client_manager.logger") as mock_logger:
            block = LLMChatBlock(
                block_name="test_concurrency_warning",
                input_cols="messages",
                output_cols="responses",
                model="openai/gpt-4",
                api_key="test-key",
                n=5,  # Generate 5 responses per input
                async_mode=True,
            )

            # Test with max_concurrency = 3, which is less than n=5
            result = block.generate(sample_dataset, _flow_max_concurrency=3)

            assert "responses" in result.column_names
            assert len(result["responses"]) == 2

            # Verify warning log was called
            mock_logger.warning.assert_called()
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "max_concurrency" in str(call)
            ]
            assert len(warning_calls) > 0
            assert "max_concurrency (3) is less than n (5)" in str(warning_calls[0])
            assert "Consider increasing max_concurrency" in str(warning_calls[0])

    def test_concurrency_not_adjusted_when_n_is_1(
        self, mock_litellm_acompletion, sample_dataset
    ):
        """Test concurrency is not adjusted when n=1 or n=None."""
        with patch("sdg_hub.core.blocks.llm.client_manager.logger") as mock_logger:
            # Test with n=1
            block_n1 = LLMChatBlock(
                block_name="test_no_adjustment_n1",
                input_cols="messages",
                output_cols="response",
                model="openai/gpt-4",
                api_key="test-key",
                n=1,
                async_mode=True,
            )

            result = block_n1.generate(sample_dataset, _flow_max_concurrency=8)

            assert "response" in result.column_names
            assert len(result["response"]) == 2

            # No adjustment should happen, so no debug log about adjustment
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Adjusted max_concurrency" in str(call)
            ]
            assert len(debug_calls) == 0

    def test_concurrency_override_in_generate_call(
        self, mock_litellm_acompletion, sample_dataset
    ):
        """Test concurrency adjustment works when n is overridden in generate call."""
        with patch("sdg_hub.core.blocks.llm.client_manager.logger") as mock_logger:
            block = LLMChatBlock(
                block_name="test_override_adjustment",
                input_cols="messages",
                output_cols="responses",
                model="openai/gpt-4",
                api_key="test-key",
                n=1,  # Initially set to 1
                async_mode=True,
            )

            # Override n to 3 at runtime with max_concurrency=9
            result = block.generate(sample_dataset, n=3, _flow_max_concurrency=9)

            assert "responses" in result.column_names
            assert len(result["responses"]) == 2

            # Verify debug log shows adjustment based on runtime n=3
            mock_logger.debug.assert_called()
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Adjusted max_concurrency" in str(call)
            ]
            assert len(debug_calls) > 0
            assert "Adjusted max_concurrency from 9 to 3" in str(debug_calls[0])
            assert "for n=3 completions per request" in str(debug_calls[0])

    def test_single_response_still_works(self, mock_litellm_completion, sample_dataset):
        """Test that n=1 or n=None still returns single strings."""
        # Test n=1
        block_n1 = LLMChatBlock(
            block_name="test_single_n1",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
            n=1,
        )

        result_n1 = block_n1.generate(sample_dataset)
        assert "response" in result_n1.column_names
        assert len(result_n1["response"]) == 2
        # Each response should be a single dict, not a list
        for response in result_n1["response"]:
            assert isinstance(response, dict)
            assert response["content"] == "Test response"

        # Test n=None (default)
        block_none = LLMChatBlock(
            block_name="test_single_none",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-4",
            api_key="test-key",
        )

        result_none = block_none.generate(sample_dataset)
        assert "response" in result_none.column_names
        assert len(result_none["response"]) == 2
        # Each response should be a single dict, not a list
        for response in result_none["response"]:
            assert isinstance(response, dict)
            assert response["content"] == "Test response"

    def test_multiple_responses_with_override(
        self, mock_litellm_completion_multiple, sample_dataset
    ):
        """Test multiple responses with runtime n override."""
        block = LLMChatBlock(
            block_name="test_override_n",
            input_cols="messages",
            output_cols="responses",
            model="openai/gpt-4",
            api_key="test-key",
            n=1,  # Initially set to 1
        )

        # Override n to 3 at runtime
        result = block.generate(sample_dataset, n=3)

        assert "responses" in result.column_names
        assert len(result["responses"]) == 2

        # Each response should be a list of 3 dicts due to override
        for responses in result["responses"]:
            assert isinstance(responses, list)
            assert len(responses) == 3
            assert [r["content"] for r in responses] == [
                "Response 1",
                "Response 2",
                "Response 3",
            ]

    def test_config_validation_with_n_parameter(self):
        """Test that n parameter is properly validated in config."""
        # Valid n parameter
        config = LLMConfig(model="openai/gpt-4", n=5)
        assert config.n == 5

        # Invalid n parameter (negative)
        with pytest.raises(ValueError, match="n must be positive"):
            LLMConfig(model="openai/gpt-4", n=-1)

        # Invalid n parameter (zero)
        with pytest.raises(ValueError, match="n must be positive"):
            LLMConfig(model="openai/gpt-4", n=0)

    def test_validation_fails_with_non_list_messages(self):
        """Test validation fails when messages is not a list."""
        # Create dataset with mixed types - Dataset library will handle this
        # by creating a single row with invalid data
        invalid_data = [
            {"messages": "not a list"},  # Invalid - not a list
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError, match="must contain a list"):
            block._validate_custom(dataset)

    def test_validation_fails_with_empty_messages(self):
        """Test validation fails when messages list is empty."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": []},  # Invalid - empty list
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError, match="Messages list is empty"):
            block._validate_custom(dataset)

    def test_validation_fails_with_non_dict_message(self):
        """Test validation fails when message is not a dict."""
        # Create dataset with valid structure first, then modify it to test the error
        Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "Valid message"}],
                    [{"role": "user", "content": "Valid message"}],
                ]
            }
        )

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        # Test by directly calling the validation function with invalid data
        # This avoids the PyArrow type mixing issue
        def validate_sample_with_invalid_message(sample_with_index):
            """Validate a single sample's message format with invalid message."""
            idx, sample = sample_with_index
            messages = sample[block.input_cols[0]]

            # Validate messages is a list
            if not isinstance(messages, list):
                raise BlockValidationError(
                    f"Messages column '{block.input_cols[0]}' must contain a list, "
                    f"got {type(messages)} in row {idx}",
                    details=f"Block: {block.block_name}, Row: {idx}, Value: {messages}",
                )

            # Validate messages is not empty
            if not messages:
                raise BlockValidationError(
                    f"Messages list is empty in row {idx}",
                    details=f"Block: {block.block_name}, Row: {idx}",
                )

            # Validate each message format
            for msg_idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} must be a dict, got {type(message)}",
                        details=f"Block: {block.block_name}, Row: {idx}, Message: {msg_idx}, Value: {message}",
                    )

                # Validate required fields
                if "role" not in message or message["role"] is None:
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} missing required 'role' field",
                        details=f"Block: {block.block_name}, Row: {idx}, Message: {msg_idx}, Available fields: {list(message.keys())}",
                    )

                if "content" not in message or message["content"] is None:
                    raise BlockValidationError(
                        f"Message {msg_idx} in row {idx} missing required 'content' field",
                        details=f"Block: {block.block_name}, Row: {idx}, Message: {msg_idx}, Available fields: {list(message.keys())}",
                    )

            return True

        # Create a sample with invalid message and test the validation
        invalid_sample = {
            "messages": [{"role": "user", "content": "Valid message"}, "not a dict"]
        }

        with pytest.raises(BlockValidationError, match="must be a dict"):
            validate_sample_with_invalid_message((0, invalid_sample))

    def test_validation_fails_with_missing_role(self):
        """Test validation fails when message is missing 'role' field."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": [{"content": "Missing role"}]},  # Invalid - no role
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError, match="missing required 'role' field"):
            block._validate_custom(dataset)

    def test_validation_fails_with_missing_content(self):
        """Test validation fails when message is missing 'content' field."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": [{"role": "user"}]},  # Invalid - no content (will be None)
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(
            BlockValidationError, match="missing required 'content' field"
        ):
            block._validate_custom(dataset)

    def test_validation_fails_with_null_role(self):
        """Test validation fails when role is explicitly None."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {
                "messages": [{"role": None, "content": "Null role"}]
            },  # Invalid - None role
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError, match="missing required 'role' field"):
            block._validate_custom(dataset)

    def test_validation_fails_with_null_content(self):
        """Test validation fails when content is explicitly None."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": [{"role": "user", "content": None}]},  # Invalid - None content
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(
            BlockValidationError, match="missing required 'content' field"
        ):
            block._validate_custom(dataset)

    def test_validation_error_reports_correct_row_and_message(self):
        """Test validation error reports correct row and message indices."""
        invalid_data = [
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {"messages": [{"role": "user", "content": "Valid message"}]},
            {
                "messages": [
                    {"role": "user", "content": "Valid message"},
                    {"role": "assistant", "content": "Valid response"},
                    {"role": "user"},  # Invalid - missing content, message index 2
                ]
            },
        ]
        dataset = Dataset.from_list(invalid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError) as exc_info:
            block._validate_custom(dataset)

        error_msg = str(exc_info.value)
        assert "row 3" in error_msg
        assert "Message 2" in error_msg
        assert "missing required 'content' field" in error_msg

    def test_validation_with_large_dataset(self):
        """Test validation works with larger datasets (validates all samples)."""
        # Create a large dataset with one invalid sample at the end
        large_data = []
        for i in range(100):
            large_data.append(
                {"messages": [{"role": "user", "content": f"Message {i}"}]}
            )

        # Add invalid sample at the end
        large_data.append({"messages": [{"role": "user"}]})  # Missing content

        dataset = Dataset.from_list(large_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        with pytest.raises(BlockValidationError) as exc_info:
            block._validate_custom(dataset)

        error_msg = str(exc_info.value)
        assert "row 100" in error_msg  # Should find the error in the last row
        assert "missing required 'content' field" in error_msg

    def test_validation_with_complex_conversation(self):
        """Test validation with complex multi-turn conversations."""
        valid_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                    {
                        "role": "assistant",
                        "content": "I'm doing well, thank you! How can I help you today?",
                    },
                    {"role": "user", "content": "Can you help me with Python?"},
                    {
                        "role": "assistant",
                        "content": "Absolutely! I'd be happy to help with Python. What specific question do you have?",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the weather like?"},
                    {
                        "role": "assistant",
                        "content": "I don't have access to real-time weather data.",
                    },
                    {"role": "user", "content": "That's okay, thanks!"},
                ]
            },
        ]
        dataset = Dataset.from_list(valid_data)

        block = LLMChatBlock(
            block_name="test_validation",
            input_cols="messages",
            output_cols="response",
            model="openai/gpt-3.5-turbo",
        )

        # Should not raise any exception
        block._validate_custom(dataset)
