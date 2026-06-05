// Shape mirrors scripts/generate_summary.py output. Hand-maintained;
// no codegen because the schema is small and stable.

export interface SummaryMetadata {
  generated_at: string;
  n_prompts_total: number;
  n_matched_subset: number;
  alpha: number;
  validator_threshold: number;
}

export interface HeadlineMetrics {
  compression: number;
  prompt_similarity: number;
  output_similarity: number;
  judge_score: number;
  persona_preserved: number;
  compression_efficiency: number;
  gate_pass_rate: number;
  constraint_preservation: number | null;
}

export interface ScatterPoint {
  id: string;
  label: string;
  domain: string;
  word_count: number;
  ours: { compression: number; output_similarity: number; judge_score: number | null; persona_preserved: boolean };
  llmlingua: { compression: number; output_similarity: number; judge_score: number | null; persona_preserved: boolean };
}

export interface DomainRow {
  domain: string;
  n: number;
  ours: { compression: number; output_similarity: number; judge_score: number | null };
  llmlingua: { compression: number; output_similarity: number; judge_score: number | null };
}

export interface DemoPrompt {
  id: string;
  category: 'ours_wins' | 'preserved' | 'llmlingua_breaks' | 'balanced';
  caption: string;
  label: string;
  domain: string;
  word_count: number;
  original_text: string;
  ours: {
    compressed_text: string;
    compression: number;
    output_similarity: number;
    judge_score: number | null;
    tier: string;
    density: number;
    persona_preserved: boolean;
    gate_passed: boolean;
  };
  llmlingua: {
    compressed_text: string;
    compression: number;
    output_similarity: number;
    judge_score: number | null;
    persona_preserved: boolean;
  };
}

export interface ExampleEntry extends Omit<DemoPrompt, 'id' | 'category' | 'caption'> {}

export interface Summary {
  metadata: SummaryMetadata;
  headline: { ours: HeadlineMetrics; llmlingua: HeadlineMetrics };
  tier_distribution: Record<string, number>;
  scatter_data: ScatterPoint[];
  domain_breakdown: DomainRow[];
  demo_prompts: DemoPrompt[];
  all_examples: Record<string, ExampleEntry>;
}

// Live API contract — mirrors CompressionResult.to_dict()
export interface LiveCompressionResult {
  original_text: string;
  compressed_text: string;
  original_tokens: number;
  compressed_tokens: number;
  semantic_similarity: number;
  gate_passed: boolean;
  validator_failures: string[];
  density: number;
  tier: number;
  best_score: number | null;
  n_evaluations: number | null;
  alpha: number | null;
  time_seconds: number;
  // Properties:
  compression_ratio: number;
  tokens_saved: number;
  compression_efficiency: number;
  tier_label: string;
  persona_preserved: boolean;
  placeholders_preserved: boolean;
  safe_to_use: boolean;
  components_original?: Record<string, unknown>;
  components_compressed?: Record<string, unknown>;
  components_text?: Record<string, unknown>;
}

export type LiveCompressionResponse =
  | { ok: true; result: LiveCompressionResult }
  | { ok: false; error: string };
