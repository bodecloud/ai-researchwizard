export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL'
}

export enum LogCategory {
  API_CALL = 'API_CALL',
  WEBSOCKET = 'WEBSOCKET',
  AGENT_WORK = 'AGENT_WORK',
  CONNECTION = 'CONNECTION',
  BACKEND_QUERY = 'BACKEND_QUERY',
  RESEARCH_PROGRESS = 'RESEARCH_PROGRESS',
  USER_INTERACTION = 'USER_INTERACTION',
  SYSTEM = 'SYSTEM'
}

type JsonPrimitive = string | number | boolean | null;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
type LogPayload = JsonValue | { [key: string]: JsonValue };
type ErrorLike = Error | { message?: string; [key: string]: JsonValue };

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  category: LogCategory;
  message: string;
  data?: LogPayload;
  requestId?: string;
  sessionId?: string;
  stack?: string;
}

class TerminalLogger {
  private static instance: TerminalLogger;
  private sessionId: string;
  private isEnabled: boolean = true;

  private constructor() {
    this.sessionId = this.generateSessionId();
    this.logSystemInfo();
  }

  public static getInstance(): TerminalLogger {
    if (!TerminalLogger.instance) {
      TerminalLogger.instance = new TerminalLogger();
    }
    return TerminalLogger.instance;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private logSystemInfo(): void {
    console.log('\n' + '='.repeat(80));
    console.log('🚀 GPT-RESEARCHER FRONTEND TERMINAL LOGGING INITIALIZED');
    console.log('='.repeat(80));
    console.log(`📅 Session: ${this.sessionId}`);
    console.log(`🕐 Started: ${new Date().toISOString()}`);
    console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log('='.repeat(80) + '\n');
  }

  private formatLog(entry: LogEntry): string {
    const timestamp = entry.timestamp;
    const level = entry.level.padEnd(8);
    const category = entry.category.padEnd(18);
    const sessionPrefix = `[${this.sessionId.slice(-8)}]`;
    const requestPrefix = entry.requestId ? `[${entry.requestId.slice(-8)}]` : '[--------]';

    return `${timestamp} ${level} ${category} ${sessionPrefix} ${requestPrefix} ${entry.message}`;
  }

  private log(level: LogLevel, category: LogCategory, message: string, data?: LogPayload, requestId?: string): void {
    if (!this.isEnabled) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      category,
      message,
      data,
      requestId,
      sessionId: this.sessionId,
      stack: level === LogLevel.ERROR || level === LogLevel.CRITICAL ? new Error().stack : undefined
    };

    const formattedMessage = this.formatLog(entry);

    // Output to terminal based on log level
    switch (level) {
      case LogLevel.DEBUG:
        console.debug(formattedMessage);
        break;
      case LogLevel.INFO:
        console.log(formattedMessage);
        break;
      case LogLevel.WARN:
        console.warn(formattedMessage);
        break;
      case LogLevel.ERROR:
        console.error(formattedMessage);
        break;
      case LogLevel.CRITICAL:
        console.error('🚨 ' + formattedMessage);
        break;
    }

    // Output data if provided
    if (data) {
      console.log(`${' '.repeat(60)}📊 DATA:`, JSON.stringify(data, null, 2));
    }

    // Output stack trace for errors
    if (entry.stack && (level === LogLevel.ERROR || level === LogLevel.CRITICAL)) {
      console.log(`${' '.repeat(60)}📋 STACK:`, entry.stack);
    }
  }

  // API Call Logging
  public logApiCall(method: string, endpoint: string, payload?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.API_CALL, `${method} ${endpoint}`, payload, requestId);
  }

  public logApiResponse(endpoint: string, status: number, responseData?: LogPayload, requestId?: string): void {
    const level = status >= 400 ? LogLevel.ERROR : LogLevel.INFO;
    this.log(level, LogCategory.API_CALL, `Response ${status} from ${endpoint}`, responseData, requestId);
  }

  public logApiError(endpoint: string, error: ErrorLike, requestId?: string): void {
    this.log(LogLevel.ERROR, LogCategory.API_CALL, `API Error at ${endpoint}: ${error instanceof Error ? error.message : (error.message ?? 'Unknown error')}`, error, requestId);
  }

  // WebSocket Logging
  public logWebSocketConnection(url: string, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.WEBSOCKET, `🔌 Connecting to WebSocket: ${url}`, undefined, requestId);
  }

  public logWebSocketConnected(url: string, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.WEBSOCKET, `✅ WebSocket connected: ${url}`, undefined, requestId);
  }

  public logWebSocketMessage(direction: 'SEND' | 'RECEIVE', message: LogPayload, requestId?: string): void {
    const icon = direction === 'SEND' ? '📤' : '📥';
    this.log(LogLevel.INFO, LogCategory.WEBSOCKET, `${icon} WebSocket ${direction}`, message, requestId);
  }

  public logWebSocketError(error: ErrorLike, requestId?: string): void {
    this.log(LogLevel.ERROR, LogCategory.WEBSOCKET, `❌ WebSocket Error: ${error instanceof Error ? error.message : (error.message ?? 'Unknown error')}`, error, requestId);
  }

  public logWebSocketClosed(code: number, reason: string, requestId?: string): void {
    this.log(LogLevel.WARN, LogCategory.WEBSOCKET, `🔌 WebSocket Closed: ${code} - ${reason}`, undefined, requestId);
  }

  // Agent Work Logging
  public logAgentWork(agentType: string, task: string, progress?: number, requestId?: string): void {
    const progressText = progress !== undefined ? ` (${Math.round(progress)}%)` : '';
    this.log(LogLevel.INFO, LogCategory.AGENT_WORK, `🤖 ${agentType}: ${task}${progressText}`, undefined, requestId);
  }

  public logAgentComplete(agentType: string, result: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.AGENT_WORK, `✅ ${agentType} completed`, result, requestId);
  }

  public logAgentError(agentType: string, error: ErrorLike, requestId?: string): void {
    this.log(LogLevel.ERROR, LogCategory.AGENT_WORK, `❌ ${agentType} error: ${error instanceof Error ? error.message : (error.message ?? 'Unknown error')}`, error, requestId);
  }

  // Connection Logging
  public logConnectionStatus(status: string, quality?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.CONNECTION, `🔗 Connection status: ${status}`, quality, requestId);
  }

  public logConnectionRetry(attempt: number, maxAttempts: number, requestId?: string): void {
    this.log(LogLevel.WARN, LogCategory.CONNECTION, `🔄 Connection retry ${attempt}/${maxAttempts}`, undefined, requestId);
  }

  // Backend Query Logging
  public logBackendQuery(query: string, parameters?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.BACKEND_QUERY, `🔍 Backend Query: ${query}`, parameters, requestId);
  }

  public logBackendResponse(query: string, responseTime: number, resultCount?: number, requestId?: string): void {
    const data = { responseTime, resultCount };
    this.log(LogLevel.INFO, LogCategory.BACKEND_QUERY, `📊 Query Response: ${query} (${responseTime}ms)`, data, requestId);
  }

  // Research Progress Logging
  public logResearchProgress(stage: string, progress: number, details?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.RESEARCH_PROGRESS, `📈 Research ${stage}: ${Math.round(progress)}%`, details, requestId);
  }

  public logResearchComplete(totalTime: number, resultSummary: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.RESEARCH_PROGRESS, `🎉 Research completed in ${totalTime}ms`, resultSummary, requestId);
  }

  // User Interaction Logging
  public logUserAction(action: string, data?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.USER_INTERACTION, `👤 User ${action}`, data, requestId);
  }

  // System Logging
  public logSystem(message: string, data?: LogPayload, requestId?: string): void {
    this.log(LogLevel.INFO, LogCategory.SYSTEM, `⚙️ ${message}`, data, requestId);
  }

  public logWarning(message: string, data?: LogPayload, requestId?: string): void {
    this.log(LogLevel.WARN, LogCategory.SYSTEM, `⚠️ ${message}`, data, requestId);
  }

  public logError(message: string, error?: ErrorLike, requestId?: string): void {
    this.log(LogLevel.ERROR, LogCategory.SYSTEM, `❌ ${message}`, error, requestId);
  }

  public logCritical(message: string, error?: ErrorLike, requestId?: string): void {
    this.log(LogLevel.CRITICAL, LogCategory.SYSTEM, `🚨 CRITICAL: ${message}`, error, requestId);
  }

  // Request ID generation
  public generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  // Control methods
  public enable(): void {
    this.isEnabled = true;
    this.logSystem('Terminal logging enabled');
  }

  public disable(): void {
    this.logSystem('Terminal logging disabled');
    this.isEnabled = false;
  }

  // Session info
  public getSessionId(): string {
    return this.sessionId;
  }
}

// Export singleton instance
export const terminalLogger = TerminalLogger.getInstance();

// Export convenience functions
export const logApiCall = (method: string, endpoint: string, payload?: LogPayload, requestId?: string) =>
  terminalLogger.logApiCall(method, endpoint, payload, requestId);

export const logApiResponse = (endpoint: string, status: number, responseData?: LogPayload, requestId?: string) =>
  terminalLogger.logApiResponse(endpoint, status, responseData, requestId);

export const logApiError = (endpoint: string, error: ErrorLike, requestId?: string) =>
  terminalLogger.logApiError(endpoint, error, requestId);

export const logWebSocketConnection = (url: string, requestId?: string) =>
  terminalLogger.logWebSocketConnection(url, requestId);

export const logWebSocketConnected = (url: string, requestId?: string) =>
  terminalLogger.logWebSocketConnected(url, requestId);

export const logWebSocketMessage = (direction: 'SEND' | 'RECEIVE', message: LogPayload, requestId?: string) =>
  terminalLogger.logWebSocketMessage(direction, message, requestId);

export const logWebSocketError = (error: ErrorLike, requestId?: string) =>
  terminalLogger.logWebSocketError(error, requestId);

export const logWebSocketClosed = (code: number, reason: string, requestId?: string) =>
  terminalLogger.logWebSocketClosed(code, reason, requestId);

export const logAgentWork = (agentType: string, task: string, progress?: number, requestId?: string) =>
  terminalLogger.logAgentWork(agentType, task, progress, requestId);

export const logAgentComplete = (agentType: string, result: LogPayload, requestId?: string) =>
  terminalLogger.logAgentComplete(agentType, result, requestId);

export const logAgentError = (agentType: string, error: ErrorLike, requestId?: string) =>
  terminalLogger.logAgentError(agentType, error, requestId);

export const logConnectionStatus = (status: string, quality?: LogPayload, requestId?: string) =>
  terminalLogger.logConnectionStatus(status, quality, requestId);

export const logConnectionRetry = (attempt: number, maxAttempts: number, requestId?: string) =>
  terminalLogger.logConnectionRetry(attempt, maxAttempts, requestId);

export const logBackendQuery = (query: string, parameters?: LogPayload, requestId?: string) =>
  terminalLogger.logBackendQuery(query, parameters, requestId);

export const logBackendResponse = (query: string, responseTime: number, resultCount?: number, requestId?: string) =>
  terminalLogger.logBackendResponse(query, responseTime, resultCount, requestId);

export const logResearchProgress = (stage: string, progress: number, details?: LogPayload, requestId?: string) =>
  terminalLogger.logResearchProgress(stage, progress, details, requestId);

export const logResearchComplete = (totalTime: number, resultSummary: LogPayload, requestId?: string) =>
  terminalLogger.logResearchComplete(totalTime, resultSummary, requestId);

export const logUserAction = (action: string, data?: LogPayload, requestId?: string) =>
  terminalLogger.logUserAction(action, data, requestId);

export const logSystem = (message: string, data?: LogPayload, requestId?: string) =>
  terminalLogger.logSystem(message, data, requestId);

export const logWarning = (message: string, data?: LogPayload, requestId?: string) =>
  terminalLogger.logWarning(message, data, requestId);

export const logError = (message: string, error?: ErrorLike, requestId?: string) =>
  terminalLogger.logError(message, error, requestId);

export const logCritical = (message: string, error?: ErrorLike, requestId?: string) =>
  terminalLogger.logCritical(message, error, requestId);

export const generateRequestId = () => terminalLogger.generateRequestId();
