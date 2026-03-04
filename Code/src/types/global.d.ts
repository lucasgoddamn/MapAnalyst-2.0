export {};

declare global {
  interface Window {
    MapAnalyst?: {
      getSettings?: () => any;
      lastResult?: any;
    };
    mapanalystImport?: (kind: "old" | "new" | "linked", file: File) => void;
    mapanalystExport?: (kind: "old" | "new" | "linked") => void;
    getLinks?: () => any[];
  }
}