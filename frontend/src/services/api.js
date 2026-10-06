import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export async function analyzeLeaf(imageFile) {
  const formData = new FormData();

  formData.append("file", imageFile);

  const response = await api.post(
    "/analyze",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
}

export async function approveSpray({
  crop,
  disease,
  severityLevel,
  deviceId,
}) {
  const response = await api.post(
    "/spray",
    {
      crop,
      disease,
      severity_level: severityLevel,
      device_id: deviceId,
      approved: true,
    }
  );

  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");

  return response.data;
}

export async function getMqttHealth() {
  const response = await api.get("/mqtt/health");

  return response.data;
}

export default api;