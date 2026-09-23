import 'dart:convert';
import 'dart:io';

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
