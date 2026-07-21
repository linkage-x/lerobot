// Shared GUI API singleton. Extracted so the page modules split out of App.tsx
// all talk to the same instance (the class also holds mock snapshot state, so a
// second `new DataCollectionGuiApi()` would diverge).
import { DataCollectionGuiApi } from "./api";

export const api = new DataCollectionGuiApi();
