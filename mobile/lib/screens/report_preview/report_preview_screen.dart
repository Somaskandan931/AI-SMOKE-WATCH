import 'package:flutter/material.dart';

import '../../core/services/api_service.dart';
import '../../core/services/x_share_service.dart';
import '../../models/vehicle_report.dart';
import '../../widgets/loading_indicator.dart';
import '../../widgets/report_card.dart';
import '../success/success_screen.dart';

/// FR-13/FR-15: preview the generated report and require explicit user
/// confirmation ("Report on X") before anything is opened for sharing.
/// The app itself never auto-publishes -- X/Twitter still requires the user
/// to tap its own "Post" button.
class ReportPreviewScreen extends StatefulWidget {
  final VehicleReport report;
  const ReportPreviewScreen({super.key, required this.report});

  @override
  State<ReportPreviewScreen> createState() => _ReportPreviewScreenState();
}

class _ReportPreviewScreenState extends State<ReportPreviewScreen> {
  final ApiService _api = ApiService();
  final XShareService _xShare = XShareService();
  late Future<Map<String, dynamic>> _future;

  @override
  void initState() {
    super.initState();
    _future = _generate();
  }

  Future<Map<String, dynamic>> _generate() async {
    final result = widget.report.detectionResult!;
    final data = await _api.generateReport(
      registrationNumber: widget.report.plate!.registrationNumber,
      smokeDetected: result.smokeDetected,
      smokeConfidence: result.smokeConfidence,
      latitude: widget.report.latitude,
      longitude: widget.report.longitude,
      placeName: widget.report.placeName,
      timestamp: widget.report.capturedAt,
    );
    widget.report.reportText = data['report_text'] as String;
    widget.report.authorityHandle = data['authority_handle'] as String;
    widget.report.xIntentUrl = data['x_intent_url'] as String;
    return data;
  }

  Future<void> _reportOnX() async {
    final opened = await _xShare.shareWithEvidence(
      text: widget.report.reportText!,
      intentUrl: widget.report.xIntentUrl!,
      image: widget.report.vehicleImage,
    );
    if (!mounted) return;
    if (opened) {
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => SuccessScreen(report: widget.report)),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not share the report. Please try again.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Report Preview')),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const LoadingIndicator(label: 'Preparing your report…');
          }
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Could not generate the report: ${snapshot.error}', textAlign: TextAlign.center),
              ),
            );
          }
          final data = snapshot.data!;
          final plateWarning = data['plate_warning'] as String?;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                if (plateWarning != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      children: [
                        const Icon(Icons.warning_amber_rounded, color: Colors.orange),
                        const SizedBox(width: 8),
                        Expanded(child: Text(plateWarning)),
                      ],
                    ),
                  ),
                ReportCard(
                  registration: data['registration'] as String,
                  location: data['location'] as String,
                  timestamp: data['timestamp'] as String,
                  smokeDetected: data['smoke_detected'] as bool,
                  reportText: data['report_text'] as String,
                ),
                const SizedBox(height: 20),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.edit),
                    label: const Text('Edit Details'),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.open_in_new),
                    label: const Text('Report on X'),
                    onPressed: _reportOnX,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Choose X in the share sheet. Your evidence photo is attached, and you post it yourself — nothing is published automatically.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 11),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
