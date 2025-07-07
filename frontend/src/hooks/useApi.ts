import { useState, useCallback } from 'react';

/**
 * Generic state for API operations
 */
interface ApiState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Response format for API operations
 */
type ApiResponse<T> = {
  success: boolean;
  data?: T;
  error?: string;
};

/**
 * Custom hook for API operations with loading, error, and response states
 * @returns State and API operation methods
 */
export const useApi = <T>() => {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    isLoading: false,
    error: null,
  });

  /**
   * Execute a function that returns a Promise
   * @param asyncFn - Async function to execute
   * @returns Promise that resolves to a standardized response
   */
  const execute = useCallback(async <R>(
    asyncFn: () => Promise<R>
  ): Promise<ApiResponse<R>> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const result = await asyncFn();
      setState(prev => ({ ...prev, data: result as unknown as T, isLoading: false }));
      return { success: true, data: result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '操作失败';
      setState(prev => ({ ...prev, error: errorMessage, isLoading: false }));
      return { success: false, error: errorMessage };
    }
  }, []);

  /**
   * Reset the state
   */
  const reset = useCallback(() => {
    setState({
      data: null,
      isLoading: false,
      error: null,
    });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}; 