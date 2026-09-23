import 'dart:io';
import 'package:flutter/material.dart';

import '../../core/services/camera_service.dart';
import '../../models/vehicle_report.dart';
import '../detection/detection_screen.dart';

class VehicleCaptureScreen extends StatefulWidget {
  final VehicleReport report;
  const VehicleCaptureScreen({super.key, required this.report});

  @override
  State<VehicleCaptureScreen> createState() => _VehicleCaptureScreenState();
}

class _VehicleCaptureScreenState extends State<VehicleCaptureScreen> {
  final CameraService _cameraService = CameraService();
  File? _preview;

  Future<void> _capture(bool fromCamera) async {
    final file = fromCamera ? await _cameraService.captureFromCamera() : await _cameraService.pickFromGallery();
    if (file == null) return;
    setState(() => _preview = file);
  }

  void _continue() {
    if (_preview == null) return;
    widget.report.vehicleImage = _preview;
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => DetectionScreen(report: widget.report)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Capture Vehicle')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.black12,
                  borderRadius: BorderRadius.circular(20),
                ),
                clipBehavior: Clip.antiAlias,
                child: _preview == null
                    ? const Center(
                        child: Icon(Icons.directions_car, size: 96, color: Colors.black26),
                      )
                    : Image.file(_preview!, fit: BoxFit.cover),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Make sure both the vehicle and the visible smoke are in frame.',
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                    onPressed: () => _capture(false),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Take Photo'),
                    onPressed: () => _capture(true),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _preview == null ? null : _continue,
                child: const Text('Analyze Photo'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
