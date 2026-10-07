import 'dart:convert';
import 'dart:typed_data';

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

/// Maps to `DetectResponse` (POST /api/detect) and, with the extra
/// plate/OCR fields below, to the combined `AnalyzeResponse`
/// (POST /api/analyze). /api/analyze is what the app actually calls now --
/// one upload of the vehicle photo gets vehicle+smoke detection *and*
/// plate location *and* OCR back in a single round trip, instead of
/// uploading the same photo three separate times. `fromJson` is kept for
/// the plain /detect endpoint, which still exists standalone.
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

  // --- plate location + OCR, populated only via fromAnalyzeJson ---
  final bool plateFound;
  final Uint8List? plateCropBytes;
  final String? plateMode;
  final String? plateMessage;
  final String? registrationNumber;
  final double ocrConfidence;
  final bool ocrAvailable;
  final String? ocrMessage;

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
    this.plateFound = false,
    this.plateCropBytes,
    this.plateMode,
    this.plateMessage,
    this.registrationNumber,
    this.ocrConfidence = 0.0,
    this.ocrAvailable = false,
    this.ocrMessage,
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

  factory DetectionResult.fromAnalyzeJson(Map<String, dynamic> json) {
    final base = DetectionResult.fromJson(json);
    final b64 = json['plate_cropped_image_base64'] as String?;
    return DetectionResult(
      vehicleDetected: base.vehicleDetected,
      smokeDetected: base.smokeDetected,
      vehicleConfidence: base.vehicleConfidence,
      smokeConfidence: base.smokeConfidence,
      smokeAssociatedWithVehicle: base.smokeAssociatedWithVehicle,
      reportingAllowed: base.reportingAllowed,
      detections: base.detections,
      mode: base.mode,
      message: base.message,
      plateFound: json['plate_found'] as bool? ?? false,
      plateCropBytes: (b64 != null && b64.isNotEmpty) ? base64Decode(b64) : null,
      plateMode: json['plate_mode'] as String?,
      plateMessage: json['plate_message'] as String?,
      registrationNumber: json['registration_number'] as String?,
      ocrConfidence: (json['ocr_confidence'] as num?)?.toDouble() ?? 0.0,
      ocrAvailable: json['ocr_available'] as bool? ?? false,
      ocrMessage: json['ocr_message'] as String?,
    );
  }
}
