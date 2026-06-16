import React from "react";
import { Search, Upload, Play } from "lucide-react";

interface HeaderProps {
  selectedAsset: string;
  setSelectedAsset: (ticker: string) => void;
  timeframe: string;
  setTimeframe: (tf: string) => void;
  onUploadClick: () => void;
  onPredictClick: () => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  isTraining: boolean;
  trainingProgress: number;
}

export default function Header({
  selectedAsset,
  setSelectedAsset,
  timeframe,
  setTimeframe,
  onUploadClick,
  onPredictClick,
  searchQuery,
  setSearchQuery,
  isTraining,
  trainingProgress,
}: HeaderProps) {
  const assets = [
    { name: "S&P 500", ticker: "^GSPC" },
    { name: "VN-Index", ticker: "^VNINDEX" },
    { name: "Bitcoin", ticker: "BTC-USD" },
    { name: "Ethereum", ticker: "ETH-USD" },
    { name: "Tesla", ticker: "TSLA" },
    { name: "Apple", ticker: "AAPL" },
    { name: "Gold", ticker: "GC=F" },
    { name: "Crude Oil", ticker: "CL=F" },
  ];

  const timeframes = ["1D", "1W", "1M", "3M", "6M", "1Y", "All"];

  return (
    <header className="header">
      <div className="header-left">
        <div className="search-box">
          <Search size={16} className="text-muted" />
          <input
            type="text"
            placeholder="Search asset..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <select
          className="select-dropdown"
          value={selectedAsset}
          onChange={(e) => setSelectedAsset(e.target.value)}
        >
          {assets.map((asset) => (
            <option key={asset.ticker} value={asset.ticker}>
              {asset.name} ({asset.ticker})
            </option>
          ))}
        </select>
      </div>

      <div className="header-right">
        <div className="timeframe-selector">
          {timeframes.map((tf) => (
            <button
              key={tf}
              className={`timeframe-btn ${timeframe === tf ? "active" : ""}`}
              onClick={() => setTimeframe(tf)}
            >
              {tf}
            </button>
          ))}
        </div>

        <button className="btn btn-secondary" onClick={onUploadClick}>
          <Upload size={14} />
          <span>Upload CSV</span>
        </button>

        <button 
          className="btn btn-primary" 
          onClick={onPredictClick} 
          disabled={isTraining}
        >
          <Play size={14} fill="white" />
          <span>{isTraining ? `Training (${trainingProgress}%)` : "Run Prediction"}</span>
        </button>

        <span className="status-badge green">Data Updated</span>
        <span className="status-badge blue">{isTraining ? "Retraining..." : "Model Ready"}</span>
      </div>
    </header>
  );
}
