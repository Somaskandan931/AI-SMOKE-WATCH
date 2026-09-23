import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/services/report_flow_controller.dart';
import '../../core/theme/app_theme.dart';
import '../report_preview/report_preview_screen.dart';

/// STEP 9 (spec numbering): "Location and Timestamp" — location is only
/// captured after an explicit user action (FR-11), never silently.
class LocationConfirmationScreen extends StatelessWidget {
  const LocationConfirmationScreen({super.key});

  void _goToPreview(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const ReportPreviewScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<ReportFlowController>();
    final draft = controller.draft;
    final hasLocation = draft.locationName != null;
    final isLoading = controller.status == FlowStatus.loading;

    return Scaffold(
      appBar: AppBar(title: const Text('Location & Timestamp')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Sharing your location helps the report, but it\'s optional. '
              'We only access it if you allow it here.',
              style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.location_on_outlined, color: AppColors.primary),
                        const SizedBox(width: 10),
                        Text(
                          hasLocation ? draft.locationName! : 'Not captured yet',
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        const Icon(Icons.schedule, color: AppColors.primary),
                        const SizedBox(width: 10),
                        Text(
                          draft.capturedAt?.toString() ?? 'Not set yet',
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const Spacer(),
            if (isLoading)
              const Center(child: CircularProgressIndicator(color: AppColors.primary))
            else ...[
              ElevatedButton.icon(
                onPressed: () async {
                  await controller.captureLocation();
                  if (context.mounted) _goToPreview(context);
                },
                icon: const Icon(Icons.my_location),
                label: const Text('Use My Location'),
              ),
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: () {
                  controller.skipLocation();
                  _goToPreview(context);
                },
                child: const Text('Skip Location'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
