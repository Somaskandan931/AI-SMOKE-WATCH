import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../../models/detection_result.dart';
import '../../models/license_plate.dart';
import '../constants/app_constants.dart';

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

/// Thin wrapper around the FastAPI endpoints:
///   GET  /api/health
///   POST /api/detect
///   POST /api/plate/detect
///   POST /api/plate/ocr
///   POST /api/report/generate
class ApiService {
  final String baseUrl;
  ApiService({this.baseUrl = AppConstants.apiBaseUrl});

  Future<bool> checkHealth() async {
    try {
      final res = await http.get(Uri.parse('$baseUrl/health')).timeout(AppConstants.requestTimeout);
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<DetectionResult> detect(File image) async {
    final response = await _postImage('$baseUrl/detect', image);
    return DetectionResult.fromJson(jsonDecode(response));
  }

  Future<Map<String, dynamic>> detectPlate(File image) async {
    final response = await _postImage('$baseUrl/plate/detect', image);
    return jsonDecode(response) as Map<String, dynamic>;
  }

  /// Locates the plate inside the full vehicle photo and reads it.
  /// Uses /plate/detect (locate + crop + OCR in one call); if that backend
  /// version returns no registration for a plate it did find, falls back to
  /// /plate/ocr on the same photo. When no plate is found, nothing is
  /// guessed -- the caller asks the user for a close-up or manual entry.
  Future<PlateScan> scanPlate(File image) async {
    final json = await detectPlate(image);
    final found = json['plate_found'] as bool? ?? false;

    Uint8List? crop;
    final b64 = json['cropped_image_base64'] as String?;
    if (b64 != null && b64.isNotEmpty) crop = base64Decode(b64);

    if (!found) {
      return PlateScan(
        plate: LicensePlate(registrationNumber: '', confidence: 0, ocrAvailable: true),
        plateFound: false,
      );
    }

    final reg = (json['registration_number'] as String?) ?? '';
    if (reg.isNotEmpty) {
      return PlateScan(
        plate: LicensePlate(
          registrationNumber: reg,
          confidence: (json['ocr_confidence'] as num?)?.toDouble() ?? 0.0,
          ocrAvailable: json['ocr_available'] as bool? ?? true,
        ),
        cropBytes: crop,
        plateFound: true,
      );
    }

    final plate = await ocrPlate(image);
    return PlateScan(plate: plate, cropBytes: crop, plateFound: true);
  }

  Future<LicensePlate> ocrPlate(File image) async {
    final response = await _postImage('$baseUrl/plate/ocr', image);
    return LicensePlate.fromOcrJson(jsonDecode(response) as Map<String, dynamic>);
  }

  Future<Map<String, dynamic>> generateReport({
    required String registrationNumber,
    required bool smokeDetected,
    required double smokeConfidence,
    double? latitude,
    double? longitude,
    String? placeName,
    DateTime? timestamp,
  }) async {
    final res = await http
        .post(
          Uri.parse('$baseUrl/report/generate'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'registration_number': registrationNumber,
            'smoke_detected': smokeDetected,
            'smoke_confidence': smokeConfidence,
            if (latitude != null) 'latitude': latitude,
            if (longitude != null) 'longitude': longitude,
            if (placeName != null) 'place_name': placeName,
            'timestamp': (timestamp ?? DateTime.now()).toUtc().toIso8601String(),
          }),
        )
        .timeout(AppConstants.requestTimeout);

    if (res.statusCode != 200) {
      throw ApiException(_extractDetail(res.body) ?? 'Could not generate report.');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<String> _postImage(String url, File image) async {
    try {
      final request = http.MultipartRequest('POST', Uri.parse(url));
      request.files.add(await http.MultipartFile.fromPath('image', image.path));
      final streamed = await request.send().timeout(AppConstants.requestTimeout);
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        throw ApiException(_extractDetail(response.body) ?? 'Server error (${response.statusCode}).');
      }
      return response.body;
    } on SocketException {
      throw ApiException('Could not reach the backend. Check API_BASE_URL and your network.');
    }
  }

  String? _extractDetail(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] != null) return decoded['detail'].toString();
    } catch (_) {}
    return null;
  }
}