class BoundingBox {
  final double x1, y1, x2, y2;
  BoundingBox({required this.x1, required this.y1, required this.x2, required this.y2});

  factory BoundingBox.fromJson(Map<String, dynamic> json) => BoundingBox(
        x1: (json['x1'] as num).toDouble(),
        y1: (json['y1'] as num).toDouble(),
        x2: (json['x2'] as num).toDouble(),
        y2: (json['y2'] as num).toDouble(),
      );
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox box;

  Detection({required this.label, required this.confidence, required this.box});

  factory Detection.fromJson(Map<String, dynamic> json) => Detection(
        label: json['label'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        box: BoundingBox.fromJson(json['box'] as Map<String, dynamic>),
      );
}

/// Maps directly to `DetectResponse` from the backend (POST /api/detect).
class DetectionResult {
  final bool vehicleDetected;
  final bool smokeDetected;
  final double vehicleConfidence;
  final double smokeConfidence;
  final bool smokeAssociatedWithVehicle;
  final bool reportingAllowed;
  final List<Detection> detections;
  final String mode; // "model" or "mock"
  final String message;

  DetectionResult({
    required this.vehicleDetected,
    required this.smokeDetected,
    required this.vehicleConfidence,
    required this.smokeConfidence,
    required this.smokeAssociatedWithVehicle,
    required this.reportingAllowed,
    required this.detections,
    required this.mode,
    required this.message,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) => DetectionResult(
        vehicleDetected: json['vehicle_detected'] as bool,
        smokeDetected: json['smoke_detected'] as bool,
        vehicleConfidence: (json['vehicle_confidence'] as num).toDouble(),
        smokeConfidence: (json['smoke_confidence'] as num).toDouble(),
        smokeAssociatedWithVehicle: json['smoke_associated_with_vehicle'] as bool,
        reportingAllowed: json['reporting_allowed'] as bool,
        detections: (json['detections'] as List<dynamic>? ?? [])
            .map((d) => Detection.fromJson(d as Map<String, dynamic>))
            .toList(),
        mode: json['mode'] as String,
        message: json['message'] as String,
      );
}
