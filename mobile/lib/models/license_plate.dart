import 'dart:typed_data';

class LicensePlate {
  /// User-visible / user-editable registration number. Starts as whatever
  /// OCR returned (may be null if OCR failed or is unavailable -- FR-10).
  String registrationNumber;
  final double confidence;
  final bool ocrAvailable;
  final bool wasManuallyEdited;

  LicensePlate({
    required this.registrationNumber,
    required this.confidence,
    required this.ocrAvailable,
    this.wasManuallyEdited = false,
  });

  factory LicensePlate.fromOcrJson(Map<String, dynamic> json) => LicensePlate(
        registrationNumber: (json['registration_number'] as String?) ?? '',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
        ocrAvailable: json['ocr_available'] as bool? ?? false,
      );

  LicensePlate copyWithEdit(String newValue) => LicensePlate(
        registrationNumber: newValue,
        confidence: confidence,
        ocrAvailable: ocrAvailable,
        wasManuallyEdited: true,
      );
}

/// Result of automatically locating + reading the plate from the vehicle photo.
class PlateScan {
  final LicensePlate plate;
  final Uint8List? cropBytes; // cropped plate region, if the backend found one
  final bool plateFound;

  PlateScan({required this.plate, this.cropBytes, required this.plateFound});
}