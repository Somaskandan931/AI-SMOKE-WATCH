import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/services/location_service.dart';
import '../../core/theme/app_theme.dart';
import '../../models/vehicle_report.dart';
import '../../widgets/loading_indicator.dart';
import '../report_preview/report_preview_screen.dart';

/// FR-11/FR-12 + PRD section 9: capture location only with permission, show
/// a human-readable place/landmark (not raw coordinates) plus timestamp,
/// and let the user review before the report is generated.
class LocationScreen extends StatefulWidget {
  final VehicleReport report;
  const LocationScreen({super.key, required this.report});

  @override
  State<LocationScreen> createState() => _LocationScreenState();
}

class _LocationScreenState extends State<LocationScreen> {
  final LocationService _locationService = LocationService();
  bool _loading = true;
  bool _permissionDenied = false;

  @override
  void initState() {
    super.initState();
    _resolveLocation();
  }

  Future<void> _resolveLocation() async {
    setState(() {
      _loading = true;
      _permissionDenied = false;
    });

    final position = await _locationService.getCurrentPosition();
    widget.report.capturedAt = DateTime.now();

    if (position == null) {
      setState(() {
        _permissionDenied = true;
        _loading = false;
      });
      return;
    }

    widget.report.latitude = position.latitude;
    widget.report.longitude = position.longitude;
    widget.report.placeName = await _locationService.reverseGeocode(position.latitude, position.longitude);

    setState(() => _loading = false);
  }

  void _continue() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => ReportPreviewScreen(report: widget.report)),
    );
  }

  void _continueWithoutLocation() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => ReportPreviewScreen(report: widget.report)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Location & Time')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: _loading
            ? const LoadingIndicator(label: 'Getting your location…')
            : _permissionDenied
                ? _permissionDeniedState()
                : _resolvedState(),
      ),
    );
  }

  Widget _permissionDeniedState() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.location_off, size: 56, color: AppColors.textSecondary),
        const SizedBox(height: 16),
        const Text(
          'Location permission was not granted. You can still generate a report — '
          'it will just be missing the location field.',
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(onPressed: _resolveLocation, child: const Text('Try Again')),
        ),
        const SizedBox(height: 10),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton(
            onPressed: _continueWithoutLocation,
            child: const Text('Continue Without Location'),
          ),
        ),
      ],
    );
  }

  Widget _resolvedState() {
    final timestamp = DateFormat('dd/MM/yyyy, hh:mm a').format(widget.report.capturedAt!);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(children: [Icon(Icons.place, color: AppColors.primary), SizedBox(width: 8), Text('Location')]),
                const SizedBox(height: 6),
                Text(
                  widget.report.placeName ?? 'Unknown location',
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
                ),
                const SizedBox(height: 18),
                const Row(children: [Icon(Icons.schedule, color: AppColors.primary), SizedBox(width: 8), Text('Date & Time')]),
                const SizedBox(height: 6),
                Text(timestamp, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
              ],
            ),
          ),
        ),
        const Spacer(),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(onPressed: _continue, child: const Text('Continue')),
        ),
      ],
    );
  }
}
