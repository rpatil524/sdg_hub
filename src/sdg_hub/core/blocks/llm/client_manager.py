# SPDX-License-Identifier: Apache-2.0
"""Client manager for LLM operations supporting all providers via LiteLLM."""

# Standard
from typing import Any, Optional, Union
import asyncio

# Third Party
from litellm import acompletion, completion
import litellm

# Local
from ...utils.logger_config import setup_logger
from .config import LLMConfig
from .error_handler import LLMErrorHandler

logger = setup_logger(__name__)


class LLMClientManager:
    """Client manager for LLM operations using LiteLLM.

    This class provides a unified interface for calling any LLM provider
    supported by LiteLLM, with robust error handling and retry logic.

    Parameters
    ----------
    config : LLMConfig
        Configuration for the LLM client.
    error_handler : Optional[LLMErrorHandler], optional
        Custom error handler. If None, a default one will be created.
    """

    def __init__(
        self, config: LLMConfig, error_handler: Optional[LLMErrorHandler] = None
    ) -> None:
        self.config = config
        self.error_handler = error_handler or LLMErrorHandler(
            max_retries=config.max_retries
        )
        self._is_loaded = False

    def load(self) -> None:
        """Load and configure the LLM client.

        This method sets up LiteLLM configuration and validates the setup.
        """
        if self._is_loaded:
            return

        # Configure LiteLLM
        self._configure_litellm()

        # Test the configuration
        self._validate_setup()

        self._is_loaded = True

        # Only log when model is actually configured
        if self.config.model:
            logger.info(
                f"Loaded LLM client for model '{self.config.model}'",
                extra={
                    "model": self.config.model,
                    "provider": self.config.get_provider(),
                    "is_local": self.config.is_local_model(),
                    "api_base": self.config.api_base,
                },
            )

    def unload(self) -> None:
        """Unload the client and clean up resources."""
        self._is_loaded = False
        try:
            logger.info(f"Unloaded LLM client for model '{self.config.model}'")
        except Exception:
            # Ignore logging errors during cleanup to prevent issues during shutdown
            pass

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings."""
        # Set global timeout for LiteLLM
        litellm.request_timeout = self.config.timeout

        # Note: API keys are now passed directly in completion calls
        # instead of modifying environment variables for thread-safety

    def _validate_setup(self) -> None:
        """Validate that the LLM setup is working."""
        try:
            # For testing/development, skip validation if using dummy API key
            if self.config.api_key == "test-key":
                logger.debug(
                    f"Skipping validation for model '{self.config.model}' (test mode)"
                )
                return

            # TODO: Skip validation for now to avoid API calls during initialization
            # we might want to make a minimal test call
            logger.debug(
                f"Setup configured for model '{self.config.model}'. "
                f"Validation will occur on first actual call."
            )

        except Exception as e:
            logger.warning(
                f"Could not validate setup for model '{self.config.model}': {e}"
            )

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert a message to a dict."""
        if hasattr(message, "to_dict"):
            return message.to_dict()
        elif hasattr(message, "__dict__"):
            return message.__dict__
        else:
            return dict(message)

    def create_completion(
        self, messages: list[dict[str, Any]], **overrides: Any
    ) -> Union[dict, list[dict]]:
        """Create a completion using LiteLLM.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Messages in OpenAI format.
        **overrides : Any
            Runtime parameter overrides.

        Returns
        -------
        Union[dict, List[dict]]
            The completion response(s). Returns a single response when n=1 or n is None,
            returns a list of responses when n>1. Response dicts contain 'content' and may contain 'reasoning_content'.

        Raises
        ------
        Exception
            If the completion fails after all retries.
        """
        if not self._is_loaded:
            self.load()

        # Merge configuration with overrides
        final_config = self.config.merge_overrides(**overrides)
        kwargs = self._build_completion_kwargs(messages, final_config)

        # Create retry wrapper
        context = {
            "model": final_config.model,
            "provider": final_config.get_provider(),
            "message_count": len(messages),
        }

        completion_func = self.error_handler.wrap_completion(
            self._call_litellm_completion, context=context
        )

        # Make the completion call
        response = completion_func(kwargs)

        # Extract message objects from response
        # Check if n > 1 to determine return type
        n_value = final_config.n or 1
        if n_value > 1:
            return [
                self._message_to_dict(choice.message) for choice in response.choices
            ]
        else:
            return self._message_to_dict(response.choices[0].message)

    async def acreate_completion(
        self,
        messages: Union[list[dict[str, Any]], list[list[dict[str, Any]]]],
        max_concurrency: Optional[int] = None,
        **overrides: Any,
    ) -> Union[dict, list[dict]] | list[Union[dict, list[dict]]]:
        """Create async completion(s) using LiteLLM with optional concurrency control.

        Parameters
        ----------
        messages : Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
            Single message list or list of message lists.
            - For single: List[Dict[str, Any]] - returns Union[Any, List[Any]]
            - For multiple: List[List[Dict[str, Any]]] - returns List[Union[Any, List[Any]]]
        max_concurrency : Optional[int], optional
            Maximum number of concurrent requests when processing multiple messages.
            If None, all requests run concurrently.
        **overrides : Any
            Runtime parameter overrides.

        Returns
        -------
        Union[dict, List[dict], List[Union[dict, List[dict]]]]
            For single message: completion response (dict when n=1, List[dict] when n>1)
            For multiple messages: list of completion responses (each element can be dict or List[dict])

        Raises
        ------
        Exception
            If the completion fails after all retries.
        """
        # Detect if we have single message or multiple messages
        if not messages:
            raise ValueError("messages cannot be empty")

        # Check if first element is a dict (single message) or list (multiple messages)
        if isinstance(messages[0], dict):
            # Single message case
            return await self._acreate_single(messages, **overrides)
        else:
            # Multiple messages case
            messages_list = messages

            if max_concurrency is not None:
                if max_concurrency < 1:
                    raise ValueError(
                        "max_concurrency must be greater than 0, got {max_concurrency}"
                    )
                # Adjust concurrency based on n parameter to avoid overwhelming API
                # when n > 1 (multiple completions per request)
                n_value = overrides.get("n") or self.config.n or 1
                if n_value > 1:
                    # Warn if max_concurrency is less than n
                    if max_concurrency < n_value:
                        logger.warning(
                            f"max_concurrency ({max_concurrency}) is less than n ({n_value}). "
                            f"This may result in very low concurrency. Consider increasing max_concurrency "
                            f"or reducing n for better performance."
                        )

                    # Reduce concurrency when generating multiple completions per request
                    adjusted_concurrency = max(1, max_concurrency // n_value)
                    logger.debug(
                        f"Adjusted max_concurrency from {max_concurrency} to {adjusted_concurrency} "
                        f"for n={n_value} completions per request"
                    )
                else:
                    adjusted_concurrency = max_concurrency

                # Use semaphore for concurrency control
                semaphore = asyncio.Semaphore(adjusted_concurrency)

                async def _create_with_semaphore(msgs):
                    async with semaphore:
                        return await self._acreate_single(msgs, **overrides)

                tasks = [_create_with_semaphore(msgs) for msgs in messages_list]
                return await asyncio.gather(*tasks)
            else:
                # No concurrency limit - process all at once
                tasks = [
                    self._acreate_single(msgs, **overrides) for msgs in messages_list
                ]
                return await asyncio.gather(*tasks)

    async def _acreate_single(
        self, messages: list[dict[str, Any]], **overrides: Any
    ) -> Union[dict, list[dict]]:
        """Create a single async completion using LiteLLM.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Messages in OpenAI format.
        **overrides : Any
            Runtime parameter overrides.

        Returns
        -------
        Union[dict, List[dict]]
            List of completion message objects. Each element is a dict when n=1 or n is None,
            or a list of dicts when n>1. Message dicts contain 'content' and may contain 'reasoning_content'.
        Raises
        ------
        Exception
            If the completion fails after all retries.
        """
        if not self._is_loaded:
            self.load()

        # Merge configuration with overrides
        final_config = self.config.merge_overrides(**overrides)
        kwargs = self._build_completion_kwargs(messages, final_config)

        # Create retry wrapper for async
        context = {
            "model": final_config.model,
            "provider": final_config.get_provider(),
            "message_count": len(messages),
        }

        completion_func = self.error_handler.wrap_completion(
            self._call_litellm_acompletion, context=context
        )

        # Make the async completion call
        response = await completion_func(kwargs)

        # Extract message objects from response
        # Check if n > 1 to determine return type
        n_value = final_config.n or 1
        if n_value > 1:
            return [
                self._message_to_dict(choice.message) for choice in response.choices
            ]
        else:
            return self._message_to_dict(response.choices[0].message)

    def create_completions_batch(
        self, messages_list: list[list[dict[str, Any]]], **overrides: Any
    ) -> list[Union[dict, list[dict]]]:
        """Create multiple completions in batch.

        Parameters
        ----------
        messages_list : List[List[Dict[str, Any]]]
            List of message lists to process.
        **overrides : Any
            Runtime parameter overrides.

        Returns
        -------
        List[dict] | List[List[dict]]
            List of completion responses. Each element is a dict when n=1 or n is None,
            or a list of dicts when n>1. Response dicts contain 'content' and may contain 'reasoning_content'.
        """
        results = []
        for messages in messages_list:
            result = self.create_completion(messages, **overrides)
            results.append(result)
        return results

    def _build_completion_kwargs(
        self, messages: list[dict[str, Any]], config: LLMConfig
    ) -> dict[str, Any]:
        """Build kwargs for LiteLLM completion call.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Messages in OpenAI format.
        config : LLMConfig
            Final configuration after merging overrides.

        Returns
        -------
        Dict[str, Any]
            Kwargs for litellm.completion().
        """
        kwargs = {
            "model": config.model,
            "messages": messages,
        }

        # Add API configuration
        if config.api_key:
            kwargs["api_key"] = config.api_key

        if config.api_base:
            kwargs["api_base"] = config.api_base

        # Add generation parameters
        generation_kwargs = config.get_generation_kwargs()
        kwargs.update(generation_kwargs)

        return kwargs

    def _call_litellm_completion(self, kwargs: dict[str, Any]) -> Any:
        """Call LiteLLM completion with error handling.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Arguments for litellm.completion().

        Returns
        -------
        Any
            LiteLLM completion response.
        """
        logger.debug(
            f"Calling LiteLLM completion for model '{kwargs['model']}'",
            extra={
                "model": kwargs["model"],
                "message_count": len(kwargs["messages"]),
                "generation_params": {
                    k: v
                    for k, v in kwargs.items()
                    if k in ["temperature", "max_tokens", "top_p", "n"]
                },
            },
        )

        response = completion(**kwargs)

        logger.debug(
            f"LiteLLM completion successful for model '{kwargs['model']}'",
            extra={
                "model": kwargs["model"],
                "choices_count": len(response.choices),
            },
        )

        return response

    async def _call_litellm_acompletion(self, kwargs: dict[str, Any]) -> Any:
        """Call LiteLLM async completion with error handling.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Arguments for litellm.acompletion().

        Returns
        -------
        Any
            LiteLLM completion response.
        """
        logger.debug(
            f"Calling LiteLLM async completion for model '{kwargs['model']}'",
            extra={
                "model": kwargs["model"],
                "message_count": len(kwargs["messages"]),
            },
        )

        response = await acompletion(**kwargs)

        logger.debug(
            f"LiteLLM async completion successful for model '{kwargs['model']}'",
            extra={
                "model": kwargs["model"],
                "choices_count": len(response.choices),
            },
        )

        return response

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model.

        Returns
        -------
        Dict[str, Any]
            Model information.
        """
        return {
            "model": self.config.model,
            "provider": self.config.get_provider(),
            "model_name": self.config.get_model_name(),
            "is_local": self.config.is_local_model(),
            "api_base": self.config.api_base,
            "is_loaded": self._is_loaded,
        }

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMClientManager(model='{self.config.model}', "
            f"provider='{self.config.get_provider()}', loaded={self._is_loaded})"
        )
