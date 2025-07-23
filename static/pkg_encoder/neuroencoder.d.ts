/* tslint:disable */
/* eslint-disable */
export function read_npz_file_encoder(data: Uint8Array): void;
export function read_model_tensors(weights_data: Uint8Array, config_string: string, tokenizer_bytes: Uint8Array, title_embeddings_data: Uint8Array, adjustments: any, aligner_weights: Uint8Array, decoder_weights: Uint8Array): void;
export function text_query(query: string): QueryResult;
export class QueryResult {
  private constructor();
  free(): void;
  readonly surface: Float32Array;
  readonly volume: Float32Array;
  readonly puborder: Float32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly read_npz_file_encoder: (a: number, b: number) => [number, number];
  readonly read_model_tensors: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: any, j: number, k: number, l: number, m: number) => [number, number];
  readonly __wbg_queryresult_free: (a: number, b: number) => void;
  readonly queryresult_surface: (a: number) => any;
  readonly queryresult_volume: (a: number) => any;
  readonly queryresult_puborder: (a: number) => any;
  readonly text_query: (a: number, b: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
