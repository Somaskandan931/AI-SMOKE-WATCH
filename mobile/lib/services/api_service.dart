import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../core/constants/api_constants.dart';
import '../models/detection_result.dart';
import '../models/license_plate.dart';
import '../models/vehicle_report.dart';

/// Thrown for any backend/network failure the UI should surface with a
/// human-readable message (spec: "Error messages should explain why
/// reporting cannot continue").
class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, {this.statusCode});

  @override
  String toString() => message;
}

class ApiService {
  final http.Client _client;
  final Duration timeout;

  ApiService({http.Client? client, this.timeout = const Duration(seconds: 30)})
      : _client = client ?? http.Client();

  Future<bool> checkHealth() async {
    try {
      final resp = await _client.get(Uri.parse(ApiConstants.health)).timeout(timeout);
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<DetectionResult> detectSmoke(Uint8List imageBytes, {String filename = 'vehicle.jpg'}) async {
    final json = await _postImage(ApiConstants.detect, imageBytes, filename);
    return DetectionResult.fromJson(json);
  }

  Future<LicensePlateResult> ocrPlate(Uint8List imageBytes, {String filename = 'plate.jpg'}) async {
    final json = await _postImage(ApiConstants.plateOcr, imageBytes, filename);
    return LicensePlateResult.fromJson(json);
  }

  Future<VehicleReport> generateReport({
    required String registrationNumber,
    required bool smokeDetected,
    required double smokeConfidence,
    double? latitude,
    double? longitude,
    String? locationName,
    String? timestamp,
  }) async {
    final resp = await _client
        .post(
          Uri.parse(ApiConstants.reportGenerate),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'registration_number': registrationNumber,
            'smoke_detected': smokeDetected,
            'smoke_confidence': smokeConfidence,
            if (latitude != null) 'latitude': latitude,
            if (longitude != null) 'longitude': longitude,
            if (locationName != null) 'location_name': locationName,
            if (timestamp != null) 'timestamp': timestamp,
          }),
        )
        .timeout(timeout);

    final json = _decodeOrThrow(resp);
    return VehicleReport.fromJson(json);
  }

  /// Uploads raw image bytes as multipart form data. Using bytes (rather
  /// than a dart:io File path) is what makes this work identically on
  /// Flutter Web, mobile, and desktop — image_picker only hands back a
  /// real filesystem path on native platforms.
  Future<Map<String, dynamic>> _postImage(String url, Uint8List imageBytes, String filename) async {
    try {
      final request = http.MultipartRequest('POST', Uri.parse(url));
      request.files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: filename));

      final streamedResponse = await _client.send(request).timeout(timeout);
      final resp = await http.Response.fromStream(streamedResponse);
      return _decodeOrThrow(resp);
    } on ApiException {
      rethrow;
    } catch (_) {
      throw ApiException(
        'Could not reach the server. Check that the backend is running and '
        'reachable, then try again.',
      );
    }
  }

  Map<String, dynamic> _decodeOrThrow(http.Response resp) {
    Map<String, dynamic> body;
    try {
      body = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('Unexpected server response (${resp.statusCode}).', statusCode: resp.statusCode);
    }

    if (resp.statusCode >= 200 && resp.statusCode < 300) {
      return body;
    }

    final detail = body['detail']?.toString() ?? 'Request failed (${resp.statusCode}).';
    throw ApiException(detail, statusCode: resp.statusCode);
  }

  void dispose() => _client.close();
}
