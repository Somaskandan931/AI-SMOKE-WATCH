import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../../models/detection_result.dart';
import '../../models/license_plate.dart';
import '../../models/vehicle_report.dart';
import '../../services/api_service.dart';
import '../../services/location_service.dart';

enum FlowStatus { idle, loading, success, error }

/// Single source of truth for the capture -> detect -> plate -> location
/// -> report flow. Screens read/mutate this via Provider rather than
/// passing a growing pile of constructor arguments down the navigator.
class ReportFlowController extends ChangeNotifier {
  final ApiService _api;
  final LocationService _locationService;

  ReportFlowController({ApiService? api, LocationService? locationService})
      : _api = api ?? ApiService(),
        _locationService = locationService ?? LocationService();

  final ReportDraft draft = ReportDraft();

  FlowStatus status = FlowStatus.idle;
  String? errorMessage;

  void _setLoading() {
    status = FlowStatus.loading;
    errorMessage = null;
    notifyListeners();
  }

  void _setError(String message) {
    status = FlowStatus.error;
    errorMessage = message;
    notifyListeners();
  }

  void _setSuccess() {
    status = FlowStatus.success;
    errorMessage = null;
    notifyListeners();
  }

  void startOver() {
    draft.reset();
    status = FlowStatus.idle;
    errorMessage = null;
    notifyListeners();
  }

  /// STEP 2-3: send the vehicle photo, get vehicle+smoke detection back.
  Future<DetectionResult?> runDetection(Uint8List imageBytes, {String filename = 'vehicle.jpg'}) async {
    draft.vehicleImageBytes = imageBytes;
    draft.vehicleImageName = filename;
    _setLoading();
    try {
      final result = await _api.detectSmoke(imageBytes, filename: filename);
      draft.detection = result;
      _setSuccess();
      return result;
    } on ApiException catch (e) {
      _setError(e.message);
      return null;
    } catch (e) {
      _setError('Something went wrong analyzing the image. Please try again.');
      return null;
    }
  }

  /// STEP 6-7: send the plate photo, get OCR text back.
  Future<LicensePlateResult?> runPlateOcr(Uint8List imageBytes, {String filename = 'plate.jpg'}) async {
    draft.plateImageBytes = imageBytes;
    draft.plateImageName = filename;
    _setLoading();
    try {
      final result = await _api.ocrPlate(imageBytes, filename: filename);
      if (!result.needsManualEntry) {
        draft.registrationNumber = result.registrationNumber;
        draft.plateConfidence = result.confidence;
      }
      _setSuccess();
      return result;
    } on ApiException catch (e) {
      _setError(e.message);
      return null;
    } catch (e) {
      _setError('Could not read the plate. You can enter it manually.');
      return null;
    }
  }

  void setManualRegistration(String value) {
    draft.registrationNumber = value.trim();
    notifyListeners();
  }

  /// STEP 7: FR-11 — only ever called from an explicit user action.
  Future<void> captureLocation() async {
    _setLoading();
    final result = await _locationService.getApproximateLocation();
    draft.latitude = result.latitude;
    draft.longitude = result.longitude;
    draft.locationName = result.label;
    draft.capturedAt = DateTime.now();
    _setSuccess();
  }

  void skipLocation() {
    draft.latitude = null;
    draft.longitude = null;
    draft.locationName = 'Location unavailable';
    draft.capturedAt = DateTime.now();
    notifyListeners();
  }

  /// STEP 8: generate the final structured report text + X intent URL.
  Future<VehicleReport?> generateReport() async {
    if (!draft.readyForReportGeneration) {
      _setError('Missing information — smoke detection and a registration number are required.');
      return null;
    }

    _setLoading();
    try {
      final report = await _api.generateReport(
        registrationNumber: draft.registrationNumber!,
        smokeDetected: draft.detection!.smokeDetected,
        smokeConfidence: draft.detection!.smokeConfidence,
        latitude: draft.latitude,
        longitude: draft.longitude,
        locationName: draft.locationName,
      );
      draft.finalReport = report;
      _setSuccess();
      return report;
    } on ApiException catch (e) {
      _setError(e.message);
      return null;
    } catch (e) {
      _setError('Could not generate the report. Please try again.');
      return null;
    }
  }

  @override
  void dispose() {
    _api.dispose();
    super.dispose();
  }
}
