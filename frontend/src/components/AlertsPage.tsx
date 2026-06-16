import React from "react";
import { AlertTriangle, AlertCircle, Info, Sparkles } from "lucide-react";

interface AlertsProps {
  alerts: any[];
  onAcknowledge: (id: string) => void;
  onResolve: (id: string) => void;
}

export default function AlertsPage({ alerts, onAcknowledge, onResolve }: AlertsProps) {
  const getIcon = (severity: string) => {
    switch (severity) {
      case "HIGH":
        return <AlertTriangle size={18} />;
      case "MEDIUM":
        return <AlertCircle size={18} />;
      default:
        return <Info size={18} />;
    }
  };

  return (
    <div className="page-container">
      {/* Alerts Feed panel */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">System Alert Feed</h3>
          <span style={{ fontSize: "12px", color: "#64748B" }}>Active and historical model signals</span>
        </div>

        <div className="alerts-list">
          {alerts.map((alert) => (
            <div className="alert-item" key={alert.id}>
              <div className="alert-item-left">
                <div className={`alert-icon-wrap ${alert.severity}`}>
                  {getIcon(alert.severity)}
                </div>
                <div className="alert-content">
                  <div className="alert-title-row">
                    <span className="alert-title">{alert.title}</span>
                    <span className="alert-asset">{alert.asset}</span>
                  </div>
                  <span className="alert-msg">{alert.message}</span>
                </div>
              </div>
              <div className="alert-item-right">
                <span className="alert-time">{alert.time}</span>
                
                {/* Status Badges */}
                <span 
                  className="status-badge"
                  style={{
                    backgroundColor: 
                      alert.status === "Active" 
                        ? "rgba(239, 68, 68, 0.15)" 
                        : alert.status === "Acknowledged"
                        ? "rgba(249, 115, 22, 0.15)"
                        : "rgba(16, 185, 129, 0.15)",
                    color: 
                      alert.status === "Active" 
                        ? "#EF4444" 
                        : alert.status === "Acknowledged"
                        ? "#F97316"
                        : "#10B981",
                  }}
                >
                  {alert.status}
                </span>

                {/* Actions */}
                <div className="alert-actions">
                  {alert.status === "Active" && (
                    <button 
                      className="btn btn-secondary" 
                      style={{ padding: "4px 10px", fontSize: "11px" }}
                      onClick={() => onAcknowledge(alert.id)}
                    >
                      Acknowledge
                    </button>
                  )}
                  {alert.status !== "Resolved" && (
                    <button 
                      className="btn btn-primary" 
                      style={{ padding: "4px 10px", fontSize: "11px" }}
                      onClick={() => onResolve(alert.id)}
                    >
                      Resolve
                    </button>
                  )}
                  <button className="btn btn-secondary" style={{ padding: "4px 10px", fontSize: "11px" }}>
                    View
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* AI market Summary at bottom */}
      <div className="panel" style={{ border: "1px solid rgba(59, 130, 246, 0.2)", background: "linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(20, 25, 35, 1) 100%)" }}>
        <div className="panel-header" style={{ marginBottom: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#3B82F6" }}>
            <Sparkles size={16} />
            <h3 className="panel-title" style={{ color: "#3B82F6" }}>AI Market Summary</h3>
          </div>
        </div>
        <p style={{ fontSize: "14px", lineHeight: "1.6", color: "#94A3B8" }}>
          Current market conditions show increased volatility across multiple assets. The S&P 500 has transitioned to a High Volatility regime with 82% confidence. Bitcoin continues to show elevated volatility at 42.8%, exceeding normal thresholds. Several assets are experiencing regime transitions, suggesting a period of market uncertainty. Recommend monitoring positions closely and adjusting risk exposure accordingly.
        </p>
      </div>
    </div>
  );
}
