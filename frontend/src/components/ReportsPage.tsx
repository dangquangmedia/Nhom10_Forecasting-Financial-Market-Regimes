import React from "react";
import { FileText, Download, Eye } from "lucide-react";

interface ReportsProps {
  reports: any[];
}

export default function ReportsPage({ reports }: ReportsProps) {
  const handleDownload = (filename: string) => {
    // Dynamically download the file from our API
    window.open(`/api/reports/${filename}/download`, "_blank");
  };

  return (
    <div className="page-container">
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Weekly Report History</h3>
          <span style={{ fontSize: "12px", color: "#64748B" }}>Generated market digest reports</span>
        </div>

        <div className="reports-list">
          {reports.map((report) => (
            <div className="report-item" key={report.id}>
              <div className="report-info">
                <FileText className="report-icon" size={24} />
                <div className="report-details">
                  <span className="report-range">{report.date_range}</span>
                  <div className="report-meta">
                    <span className="status-badge green">Generated</span>
                    <span className="report-size">{report.size}</span>
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: "10px" }}>
                <button className="btn btn-secondary">
                  <Eye size={14} />
                  <span>View</span>
                </button>
                <button 
                  className="btn btn-primary"
                  onClick={() => handleDownload(report.filename)}
                >
                  <Download size={14} />
                  <span>Download</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
