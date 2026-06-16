import React from "react";
import {
  LayoutDashboard,
  LineChart,
  Star,
  Activity,
  Brain,
  Bell,
  FileText,
  Settings as SettingsIcon,
} from "lucide-react";

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  unreadAlertsCount: number;
}

export default function Sidebar({ activeTab, setActiveTab, unreadAlertsCount }: SidebarProps) {
  const menuItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "market-regime", label: "Market Regime", icon: LineChart },
    { id: "watchlist", label: "Watchlist", icon: Star },
    { id: "forecast", label: "Forecast", icon: Activity },
    { id: "explainability", label: "Model Explainability", icon: Brain },
    { id: "alerts", label: "Alerts", icon: Bell, badge: unreadAlertsCount },
    { id: "reports", label: "Reports", icon: FileText },
    { id: "settings", label: "Settings", icon: SettingsIcon },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <h1>RegimeSense AI</h1>
        <span>Market Regime Forecaster</span>
      </div>
      <ul className="sidebar-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <li
              key={item.id}
              className={`sidebar-item ${isActive ? "active" : ""}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {item.badge && item.badge > 0 ? (
                <span
                  style={{
                    marginLeft: "auto",
                    backgroundColor: "#EF4444",
                    color: "white",
                    borderRadius: "10px",
                    padding: "2px 6px",
                    fontSize: "10px",
                    fontWeight: 700,
                  }}
                >
                  {item.badge}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
