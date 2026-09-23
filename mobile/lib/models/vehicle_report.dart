import 'dart:io';

import 'detection_result.dart';
import 'license_plate.dart';

/// Carries all state collected across the flow (Home -> Capture -> AI
/// Analysis -> Plate Capture -> OCR -> Location -> Report Preview),
/// per the PRD's step-by-step user flow (section 7).
class VehicleReport {
  File? vehicleImage;
  DetectionResult? detectionResult;

  File? plateImage;
  LicensePlate? plate;

  double? latitude;
  double? longitude;
  String? placeName; // human-readable landmark/locality (see LocationService)
  DateTime? capturedAt;

  String? reportText;
  String? authorityHandle;
  String? xIntentUrl;

  bool get canProceedPastDetection => detectionResult?.reportingAllowed ?? false;

  bool get hasRegistration =>
      plate != null && plate!.registrationNumber.trim().isNotEmpty;

  bool get hasLocation => latitude != null && longitude != null;
}
