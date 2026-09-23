import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../../core/services/api_service.dart';
import '../../models/detection_result.dart';
import '../../models/vehicle_report.dart';
import '../../widgets/detection_overlay.dart';
import '../../widgets/loading_indicator.dart';
import '../../widgets/smoke_status_card.dart';
import '../plate_capture/plate_capture_screen.dart';
import '../vehicle_capture/vehicle_capture_screen.dart';

class DetectionScreen extends StatefulWidget {
  final VehicleReport report;
  const DetectionScreen({super.key, required this.report});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  final ApiService _api = ApiService();
  late Future<DetectionResult> _future;
  ui.Image? _decodedImage;

  @override
  void initState() {
    super.initState();
    _future = _runDetection();
  }

  Future<DetectionResult> _runDetection() async {
    final bytes = await widget.report.vehicleImage!.readAsBytes();
    final image = await decodeImageFromList(Uint8List.fromList(bytes));
    _decodedImage = image;
    final result = await _api.detect(widget.report.vehicleImage!);
    widget.report.detectionResult = result;
    return result;
  }

  void _retake() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => VehicleCaptureScreen(report: VehicleReport())),
    );
  }

  void _continueToPlate() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => PlateCaptureScreen(report: widget.report)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AI Analysis')),
      body: FutureBuilder<DetectionResult>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const LoadingIndicator(label: 'Analyzing photo for vehicle and visible smoke…');
          }
          if (snapshot.hasError) {
            return _errorState(snapshot.error.toString());
          }

          final result = snapshot.data!;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                if (_decodedImage != null)
                  DetectionOverlay(
                    image: widget.report.vehicleImage!,
                    detections: result.detections,
                    imageWidth: _decodedImage!.width.toDouble(),
                    imageHeight: _decodedImage!.height.toDouble(),
                  ),
                const SizedBox(height: 20),
                SmokeStatusCard(result: result),
                const SizedBox(height: 12),
                Text(result.message, textAlign: TextAlign.center),
                const SizedBox(height: 24),
                if (result.reportingAllowed)
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _continueToPlate,
                      child: const Text('Continue to License Plate'),
                    ),
                  )
                else
                  Column(
                    children: [
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(onPressed: _retake, child: const Text('Retake Photo')),
                      ),
                      const SizedBox(height: 10),
                      SizedBox(
                        width: double.infinity,
                        child: OutlinedButton(
                          onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
                          child: const Text('Cancel'),
                        ),
                      ),
                    ],
                  ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _errorState(String message) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 48, color: Colors.redAccent),
          const SizedBox(height: 12),
          Text(message, textAlign: TextAlign.center),
          const SizedBox(height: 20),
          ElevatedButton(onPressed: _retake, child: const Text('Retake Photo')),
        ],
      ),
    );
  }
}
