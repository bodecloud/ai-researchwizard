declare module 'gpt-researcher-ui' {
  import React from 'react';

  export interface GPTResearcherProps {
    apiUrl?: string;
    apiKey?: string;
    defaultPrompt?: string;
    onResultsChange?: (results: Array<{ type: string; content: string }>) => void;
    theme?: Record<string, string | number | boolean | null | object>;
  }

  export const GPTResearcher: React.FC<GPTResearcherProps>;
}
