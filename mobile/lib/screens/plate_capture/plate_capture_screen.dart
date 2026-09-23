import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import '../../core/services/api_service.dart';
import '../../core/services/camera_service.dart';
import '../../core/theme/app_theme.dart';
import '../../models/license_plate.dart';
import '../../models/vehicle_report.dart';
import '../../widgets/loading_indicator.dart';
import '../location/location_screen.dart';

class PlateCaptureScreen extends StatefulWidget {
  final VehicleReport report;
  const PlateCaptureScreen({super.key, required this.report});

  @override
  State<PlateCaptureScreen> createState() => _PlateCaptureScreenState();
}

class _PlateCaptureScreenState extends State<PlateCaptureScreen> {
  final CameraService _cameraService = CameraService();
  final ApiService _api = ApiService();
  final TextEditingController _controller = TextEditingController();

  File? _plateImage;
  Uint8List? _plateCrop;
  bool _plateFound = true;
  bool _loading = false;
  LicensePlate? _plate;
  String? _error;

  @override
  void initState() {
    super.initState();
    _controller.addListener(() => setState(() {}));
    // The vehicle photo already contains the plate: locate and read it
    // automatically. The user can still retake a close-up or type it in.
    final vehicle = widget.report.vehicleImage;
    if (vehicle != null) {
      _loading = true;
      WidgetsBinding.instance.addPostFrameCallback((_) => _autoScan(vehicle));
    }
  }

  Future<void> _autoScan(File vehicle) async {
    try {
      final scan = await _api.scanPlate(vehicle);
      if (!mounted) return;
      setState(() {
        _plateImage = vehicle;
        _plateCrop = scan.cropBytes;
        _plateFound = scan.plateFound;
        _plate = scan.plate;
        _controller.text = scan.plate.registrationNumber;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _plateImage = vehicle;
        _error = e.toString();
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _capture(bool fromCamera) async {
    final file = fromCamera ? await _cameraService.captureFromCamera() : await _cameraService.pickFromGallery();
    if (file == null) return;
    setState(() {
      _plateImage = file;
      _plateCrop = null;
      _plateFound = true;
      _plate = null;
      _error = null;
    });
    await _runOcr(file);
  }

  Future<void> _runOcr(File file) async {
    setState(() => _loading = true);
    try {
      final plate = await _api.ocrPlate(file);
      setState(() {
        _plate = plate;
        _controller.text = plate.registrationNumber;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  void _confirm() {
    final value = _controller.text.trim();
    if (value.isEmpty) return;
    widget.report.plateImage = _plateImage;
    widget.report.plate = (_plate ?? LicensePlate(registrationNumber: '', confidence: 0, ocrAvailable: false))
        .copyWithEdit(value.toUpperCase());
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => LocationScreen(report: widget.report)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('License Plate')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(color: Colors.black12, borderRadius: BorderRadius.circular(20)),
                clipBehavior: Clip.antiAlias,
                child: _plateImage == null
                    ? const Center(child: Icon(Icons.pin, size: 80, color: Colors.black26))
                    : (_plateCrop != null
                        ? Image.memory(_plateCrop!, fit: BoxFit.contain)
                        : Image.file(_plateImage!, fit: BoxFit.cover)),
              ),
            ),
            const SizedBox(height: 16),
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
                    label: const Text('Plate Close-up'),
                    onPressed: () => _capture(true),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            if (_loading) const LoadingIndicator(label: 'Detecting license plate…'),
            if (!_loading && _plateImage != null) ...[
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(_error!, style: const TextStyle(color: AppColors.danger)),
                )
              else if (!_plateFound)
                const Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Text(
                    'Could not locate the license plate in your photo. Take a close-up of the plate, or type the registration below.',
                    style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                )
              else if (_plate != null && !_plate!.ocrAvailable)
                const Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Text(
                    'OCR isn\'t available on this server — please type the registration number manually.',
                    style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                )
              else if (_plate != null && _plate!.registrationNumber.isEmpty)
                const Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Text(
                    'Could not read the plate clearly — please enter it manually or retake the photo.',
                    style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                ),
              TextField(
                controller: _controller,
                textCapitalization: TextCapitalization.characters,
                decoration: const InputDecoration(
                  labelText: 'Detected Registration',
                  border: OutlineInputBorder(),
                  suffixIcon: Icon(Icons.edit),
                ),
              ),
              if (_plate != null && _plate!.registrationNumber.isNotEmpty)
                const Padding(
                  padding: EdgeInsets.only(top: 6),
                  child: Text(
                    'Read automatically from your photo — please verify before continuing.',
                    style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
                  ),
                ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _controller.text.trim().isEmpty ? null : _confirm,
                  child: const Text('Confirm & Continue'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}