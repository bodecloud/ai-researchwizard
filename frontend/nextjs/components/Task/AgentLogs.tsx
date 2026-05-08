type AgentLogEntry = { output: string };

interface AgentLogsProps {
  agentLogs: AgentLogEntry[];
}

export default function AgentLogs({ agentLogs }: AgentLogsProps) {
  const renderAgentLogs = (entries: AgentLogEntry[]) => {
    return entries && entries.map((agentLog: AgentLogEntry, index: number) => {
      return (<div key={index}>{agentLog.output}</div>)
    })
  }

  return (
    <div className="margin-div">
        <h2>Agent Output</h2>
        <div id="output">
          {renderAgentLogs(agentLogs)}
        </div>
    </div>
  );
}