/**
 * The message for a failed gateway command.
 *
 * The gateway always answers an error with JSON `{"error": "..."}`. A 5xx without that body did not
 * come from the gateway: it is the dev server's proxy saying nothing answered on the gateway port
 * (gateway stopped, restarting, or Thor unreachable). Showing it as a bare "HTTP 500" sent people
 * looking for a gateway bug (11-13: the manual camera recovery used to kill the gateway).
 */
export function gatewayErrorMessage(status: number, payload: unknown): string {
  if (payload && typeof payload === "object" && typeof (payload as { error?: unknown }).error === "string") {
    return (payload as { error: string }).error;
  }
  if (status >= 500 && status <= 504) {
    return `HTTP ${status}：gateway 没有应答（没在运行、正在重启或 Thor 连不上），不是 gateway 返回的错误；一直这样请找维护者`;
  }
  return `HTTP ${status}`;
}
