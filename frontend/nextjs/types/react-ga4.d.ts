declare module 'react-ga4' {
    type JsonPrimitive = string | number | boolean | null;
    type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

    export interface InitOptions {
      gaOptions?: { [key: string]: JsonValue };
      gtagOptions?: { [key: string]: JsonValue };
      testMode?: boolean;
    }
  
    export function initialize(
      measurementId: string | string[],
      options?: InitOptions
    ): void;
  
    export function event(options: {
      category: string;
      action: string;
      label?: string;
      value?: number;
      nonInteraction?: boolean;
      transport?: 'beacon' | 'xhr' | 'image';
      [key: string]: JsonValue | undefined;
    }): void;
  
    // Add other methods as needed
    export default {
      initialize,
      event
    };
  }