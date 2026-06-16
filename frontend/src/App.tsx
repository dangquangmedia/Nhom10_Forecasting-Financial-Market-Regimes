import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import OverviewPage from "./components/OverviewPage";
import MarketRegimePage from "./components/MarketRegimePage";
import WatchlistPage from "./components/WatchlistPage";
import ForecastPage from "./components/ForecastPage";
import ModelExplainabilityPage from "./components/ModelExplainabilityPage";
import AlertsPage from "./components/AlertsPage";
import ReportsPage from "./components/ReportsPage";
import SettingsPage from "./components/SettingsPage";
import { UploadCloud, X } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAsset, setSelectedAsset] = useState("^GSPC");
  const [timeframe, setTimeframe] = useState("1Y");
  const [searchQuery, setSearchQuery] = useState("");
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  
  // Model and training states
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  // API Data states
  const [watchlist, setWatchlist] = useState<any[]>([]);
  const [selectedAssetDetails, setSelectedAssetDetails] = useState<any>(null);
  const [explainabilityData, setExplainabilityData] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [reports, setReports] = useState<any[]>([]);
  const [settings, setSettings] = useState<any>({
    active_model: "XGBoost + HMM",
    confidence_threshold: 0.70,
    alert_threshold: 0.30,
    theme: "Dark Mode",
    currency: "USD",
    timezone: "UTC",
    default_assets: [],
  });

  // Fetch initial system configurations & watchlist on mount
  useEffect(() => {
    fetchWatchlist();
    fetchExplainability();
    fetchAlerts();
    fetchReports();
    fetchSettings();
  }, []);

  // Fetch detailed asset analysis whenever asset selection or timeframe filter updates
  useEffect(() => {
    fetchAssetDetails(selectedAsset, timeframe);
  }, [selectedAsset, timeframe]);

  // Polling mechanism to track retraining progress
  useEffect(() => {
    let intervalId: any;
    if (isTraining) {
      intervalId = setInterval(async () => {
        try {
          const res = await fetch("/api/retrain/status");
          const data = await res.json();
          setTrainingProgress(data.progress);
          if (!data.is_training) {
            setIsTraining(false);
            setTrainingProgress(100);
            fetchWatchlist();
            fetchAssetDetails(selectedAsset, timeframe);
            fetchExplainability();
          }
        } catch (e) {
          console.error("Error polling retraining status", e);
        }
      }, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTraining, selectedAsset, timeframe]);

  const fetchWatchlist = async () => {
    try {
      const res = await fetch("/api/assets");
      const data = await res.json();
      setWatchlist(data);
    } catch (e) {
      console.error("Failed fetching watchlist", e);
    }
  };

  const fetchAssetDetails = async (ticker: string, tf: string) => {
    try {
      const res = await fetch(`/api/asset/${ticker}?timeframe=${tf}`);
      const data = await res.json();
      setSelectedAssetDetails(data);
    } catch (e) {
      console.error("Failed fetching asset details", e);
    }
  };

  const fetchExplainability = async () => {
    try {
      const res = await fetch("/api/model-explainability");
      const data = await res.json();
      setExplainabilityData(data);
    } catch (e) {
      console.error("Failed fetching explainability data", e);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch("/api/alerts");
      const data = await res.json();
      setAlerts(data);
    } catch (e) {
      console.error("Failed fetching alerts", e);
    }
  };

  const fetchReports = async () => {
    try {
      const res = await fetch("/api/reports");
      const data = await res.json();
      setReports(data);
    } catch (e) {
      console.error("Failed fetching reports", e);
    }
  };

  const fetchSettings = async () => {
    try {
      const res = await fetch("/api/settings");
      const data = await res.json();
      setSettings(data);
    } catch (e) {
      console.error("Failed fetching settings", e);
    }
  };

  // Alert State Modifier Handlers
  const handleAcknowledgeAlert = async (id: string) => {
    try {
      const res = await fetch(`/api/alerts/${id}/acknowledge`, { method: "POST" });
      if (res.ok) {
        fetchAlerts();
      }
    } catch (e) {
      console.error("Error acknowledging alert", e);
    }
  };

  const handleResolveAlert = async (id: string) => {
    try {
      const res = await fetch(`/api/alerts/${id}/resolve`, { method: "POST" });
      if (res.ok) {
        fetchAlerts();
      }
    } catch (e) {
      console.error("Error resolving alert", e);
    }
  };

  const handleUpdateSettings = async (newSettings: any) => {
    try {
      const updated = { ...settings, ...newSettings };
      setSettings(updated);
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSettings),
      });
    } catch (e) {
      console.error("Error updating settings", e);
    }
  };

  const handleRetrain = async () => {
    try {
      const res = await fetch("/api/retrain", { method: "POST" });
      const data = await res.json();
      if (data.status === "started" || data.status === "already_running") {
        setIsTraining(true);
        setTrainingProgress(0);
      }
    } catch (e) {
      console.error("Error starting retraining", e);
    }
  };

  // CSV file uploader logic
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload-csv", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        alert(`Successfully uploaded CSV dataset: ${data.filename}\nParsed ${data.rows_parsed} rows!`);
        setIsUploadModalOpen(false);
      } else {
        alert(`Upload failed: ${data.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Error uploading CSV", err);
      alert("Network error while uploading dataset");
    }
  };

  // Filter watchlist based on searchQuery
  const filteredWatchlist = watchlist.filter((asset) =>
    asset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    asset.ticker.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const unreadAlertsCount = alerts.filter((a) => a.status === "Active").length;

  const renderActiveTabContent = () => {
    switch (activeTab) {
      case "overview":
        return (
          <OverviewPage
            data={watchlist}
            selectedAssetDetails={selectedAssetDetails}
            selectedAsset={selectedAsset}
          />
        );
      case "market-regime":
        return <MarketRegimePage />;
      case "watchlist":
        return (
          <WatchlistPage
            assetsList={filteredWatchlist}
            onAssetSelect={(ticker) => {
              setSelectedAsset(ticker);
              fetchAssetDetails(ticker, timeframe);
            }}
            selectedAssetDetails={selectedAssetDetails}
          />
        );
      case "forecast":
        return (
          <ForecastPage
            selectedAssetDetails={selectedAssetDetails}
            selectedAsset={selectedAsset}
          />
        );
      case "explainability":
        return <ModelExplainabilityPage explainabilityData={explainabilityData} />;
      case "alerts":
        return (
          <AlertsPage
            alerts={alerts}
            onAcknowledge={handleAcknowledgeAlert}
            onResolve={handleResolveAlert}
          />
        );
      case "reports":
        return <ReportsPage reports={reports} />;
      case "settings":
        return (
          <SettingsPage
            settings={settings}
            onUpdateSettings={handleUpdateSettings}
            onRetrain={handleRetrain}
            isTraining={isTraining}
            trainingProgress={trainingProgress}
          />
        );
      default:
        return <div>Tab not found</div>;
    }
  };

  return (
    <div className="app-container">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        unreadAlertsCount={unreadAlertsCount}
      />
      <div className="main-wrapper">
        <Header
          selectedAsset={selectedAsset}
          setSelectedAsset={setSelectedAsset}
          timeframe={timeframe}
          setTimeframe={setTimeframe}
          onUploadClick={() => setIsUploadModalOpen(true)}
          onPredictClick={handleRetrain}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          isTraining={isTraining}
          trainingProgress={trainingProgress}
        />
        {renderActiveTabContent()}
      </div>

      {/* Upload CSV modal overlay */}
      {isUploadModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <span className="modal-title">Upload Custom CSV Dataset</span>
              <button className="close-btn" onClick={() => setIsUploadModalOpen(false)}>
                <X size={18} />
              </button>
            </div>
            
            <label className="file-dropzone">
              <UploadCloud size={36} className="text-blue-500" />
              <p>Drag and drop or click to choose a CSV file</p>
              <span>Required columns: date, close, volume</span>
              <input
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
