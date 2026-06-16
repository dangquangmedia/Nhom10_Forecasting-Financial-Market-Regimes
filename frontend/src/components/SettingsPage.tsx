import React from "react";
import { Brain, User, Save, RefreshCw } from "lucide-react";

interface SettingsProps {
  settings: any;
  onUpdateSettings: (newSettings: any) => void;
  onRetrain: () => void;
  isTraining: boolean;
  trainingProgress: number;
}

export default function SettingsPage({
  settings,
  onUpdateSettings,
  onRetrain,
  isTraining,
  trainingProgress,
}: SettingsProps) {
  const handleSliderChange = (key: string, val: number) => {
    onUpdateSettings({ [key]: val });
  };

  const handleInputChange = (key: string, val: string) => {
    onUpdateSettings({ [key]: val });
  };

  return (
    <div className="page-container">
      <div className="settings-grid">
        {/* Model settings section */}
        <div className="panel">
          <div className="settings-section-title">
            <Brain size={16} />
            <span>Model Settings</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            <div className="form-group">
              <label className="form-label">Active Model</label>
              <select
                className="select-dropdown"
                style={{ width: "100%", backgroundColor: "var(--bg-main)" }}
                value={settings.active_model}
                onChange={(e) => handleInputChange("active_model", e.target.value)}
              >
                <option value="XGBoost + HMM">XGBoost + HMM (Primary)</option>
                <option value="Random Forest">Random Forest</option>
                <option value="Logistic Regression">Logistic Regression</option>
              </select>
            </div>

            <div className="form-group-row">
              <div className="form-group">
                <div className="form-label-row">
                  <label className="form-label">Confidence Threshold</label>
                  <span style={{ fontSize: "13px", fontWeight: 700, color: "#3B82F6" }}>
                    {(settings.confidence_threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="slider-container">
                  <input
                    type="range"
                    min="0.50"
                    max="0.95"
                    step="0.05"
                    value={settings.confidence_threshold}
                    className="range-slider"
                    onChange={(e) => handleSliderChange("confidence_threshold", parseFloat(e.target.value))}
                  />
                  <div className="slider-labels">
                    <span>50%</span>
                    <span>95%</span>
                  </div>
                </div>
              </div>

              <div className="form-group">
                <div className="form-label-row">
                  <label className="form-label">Alert Threshold (Volatility)</label>
                  <span style={{ fontSize: "13px", fontWeight: 700, color: "#F97316" }}>
                    {(settings.alert_threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="slider-container">
                  <input
                    type="range"
                    min="0.20"
                    max="0.50"
                    step="0.02"
                    value={settings.alert_threshold}
                    className="range-slider"
                    onChange={(e) => handleSliderChange("alert_threshold", parseFloat(e.target.value))}
                  />
                  <div className="slider-labels">
                    <span>20%</span>
                    <span>50%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Retrain Action */}
            <div style={{ paddingTop: "16px", borderTop: "1px solid var(--border-color)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", flex: 1 }}>
                <span style={{ fontSize: "13px", fontWeight: 600 }}>Re-train Machine Learning Models</span>
                <span style={{ fontSize: "11px", color: "#64748B" }}>Downloads online FRED/Yahoo Finance datasets and refits baseline and XGBoost models.</span>
              </div>
              
              <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                {isTraining && (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "4px" }}>
                    <span style={{ fontSize: "12px", color: "#3B82F6", fontWeight: 600 }}>Retraining...</span>
                    <div className="progress-track" style={{ width: "100px", height: "4px" }}>
                      <div className="progress-fill" style={{ width: `${trainingProgress}%`, backgroundColor: "#3B82F6" }}></div>
                    </div>
                  </div>
                )}
                <button
                  className="btn btn-primary"
                  style={{ display: "flex", gap: "8px", alignItems: "center" }}
                  onClick={onRetrain}
                  disabled={isTraining}
                >
                  <RefreshCw size={14} className={isTraining ? "animate-spin" : ""} />
                  <span>{isTraining ? "Retraining..." : "Retrain Model"}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* User Preferences section */}
        <div className="panel">
          <div className="settings-section-title">
            <User size={16} />
            <span>User Preferences</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div className="form-group-row">
              <div className="form-group">
                <label className="form-label">Theme</label>
                <select
                  className="select-dropdown"
                  style={{ width: "100%", backgroundColor: "var(--bg-main)" }}
                  value={settings.theme}
                  onChange={(e) => handleInputChange("theme", e.target.value)}
                >
                  <option value="Dark Mode">Dark Mode (Default)</option>
                  <option value="Light Mode">Light Mode</option>
                  <option value="Glassmorphic">Glassmorphic Glow</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Currency</label>
                <select
                  className="select-dropdown"
                  style={{ width: "100%", backgroundColor: "var(--bg-main)" }}
                  value={settings.currency}
                  onChange={(e) => handleInputChange("currency", e.target.value)}
                >
                  <option value="USD">USD ($)</option>
                  <option value="EUR">EUR (€)</option>
                  <option value="VND">VND (₫)</option>
                </select>
              </div>
            </div>

            <div className="form-group-row">
              <div className="form-group">
                <label className="form-label">Timezone</label>
                <select
                  className="select-dropdown"
                  style={{ width: "100%", backgroundColor: "var(--bg-main)" }}
                  value={settings.timezone}
                  onChange={(e) => handleInputChange("timezone", e.target.value)}
                >
                  <option value="UTC">UTC</option>
                  <option value="GMT+7">GMT+7 (Indochina)</option>
                  <option value="EST">EST (Eastern Time)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Default Asset Watchlist</label>
                <input
                  type="text"
                  className="form-input"
                  value={settings.default_assets.join(", ")}
                  onChange={(e) => onUpdateSettings({ default_assets: e.target.value.split(", ") })}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
