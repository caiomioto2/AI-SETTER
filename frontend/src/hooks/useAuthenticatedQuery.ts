/**
 * Authenticated query wrapper.
 * Prevents Supabase calls when session is invalid/expired,
 * avoiding the 400 retry loops seen in the dashboard.
 */
import { useQuery, type UseQueryOptions, type UseQueryResult } from '@tanstack/react-query';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';

/**
 * Returns true if the current Supabase session has a valid (non-expired) JWT.
 * Safe to call during render — reads from supabase.auth synchronously.
 */
export function hasValidSession(): boolean {
  const session = supabase.auth.getSessionSync?.();
  if (!session?.session?.access_token) return false;
  // Decode JWT payload and check exp (seconds since epoch)
  try {
    const payload = JSON.parse(atob(session.session.access_token.split('.')[1]));
    return payload.exp * 1000 > Date.now();
  } catch {
    return false;
  }
}

/**
 * Wraps useQuery with an auth guard: the query is disabled when no valid session exists.
 * This prevents 400 errors from firing on every page load when tokens are expired.
 *
 * Usage is identical to useQuery, but automatically guards authentication.
 */
export function useAuthenticatedQuery<TQueryFnData, TError = Error>(
  options: UseQueryOptions<TQueryFnData, TError> & { queryKey: readonly unknown[] }
): UseQueryResult<TQueryFnData, TError> {
  const { user, loading } = useAuth();

  // Also check JWT expiry at the token level, not just React state
  const tokenValid = !loading && user && hasValidSession();

  return useQuery({
    ...options,
    enabled: options.enabled !== false && !!tokenValid,
  });
}
