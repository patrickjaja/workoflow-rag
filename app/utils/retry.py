import asyncio
import functools
from typing import TypeVar, Callable, Optional, Union, Type
from loguru import logger
import time
from openai import RateLimitError

T = TypeVar('T')


def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (RateLimitError,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Extract retry delay from rate limit error if available
                    retry_after = None
                    if isinstance(e, RateLimitError) and hasattr(e, 'response'):
                        # Try to get retry-after header
                        headers = getattr(e.response, 'headers', {})
                        retry_after = headers.get('retry-after')
                        if retry_after:
                            try:
                                retry_after = float(retry_after)
                            except:
                                retry_after = None
                    
                    # Use retry_after if available, otherwise use exponential backoff
                    actual_delay = retry_after if retry_after else delay
                    actual_delay = min(actual_delay, max_delay)
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Rate limit error in {func.__name__} (attempt {attempt + 1}/{max_attempts}). "
                            f"Retrying in {actual_delay:.1f}s... Error: {str(e)}"
                        )
                        await asyncio.sleep(actual_delay)
                        
                        # Exponential backoff for next attempt
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"Max retries ({max_attempts}) exceeded for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
                        
                except Exception as e:
                    # For non-retryable exceptions, raise immediately
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
            
            # If we've exhausted all retries, raise the last exception
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter to prevent hitting API limits."""
    
    def __init__(self, requests_per_minute: int = 300):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.min_interval:
                wait_time = self.min_interval - time_since_last_request
                logger.debug(f"Rate limiter: waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()