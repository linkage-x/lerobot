import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null; componentStack: string | null };

// Without this, any unhandled error thrown during render unmounts the whole
// React tree and leaves a blank white page. The boundary instead shows the
// error message + stack (also logged to the console) so the failure is
// diagnosable rather than a silent white screen.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Uncaught render error:", error, info.componentStack);
    this.setState({ componentStack: info.componentStack ?? null });
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { error, componentStack } = this.state;
    if (!error) {
      return this.props.children;
    }
    return (
      <div className="error-boundary">
        <h2 className="error-boundary-title">The UI hit an unhandled error</h2>
        <p className="error-boundary-hint">
          Details below (also in the browser console). This usually means a render bug,
          not a backend failure.
        </p>
        <pre className="error-boundary-stack">{String(error.stack ?? error.message)}</pre>
        {componentStack && <pre className="error-boundary-stack">{componentStack}</pre>}
        <button className="error-boundary-reload" onClick={this.handleReload}>
          Reload
        </button>
      </div>
    );
  }
}
