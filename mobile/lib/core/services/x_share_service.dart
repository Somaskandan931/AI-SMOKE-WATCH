import 'dart:io';

import 'package:share_plus/share_plus.dart';
import 'package:url_launcher/url_launcher.dart';

/// FR-14/FR-15: hand the report to X so the USER posts it from their own
/// account -- NEVER publish automatically.
///
/// The X intent URL is text-only and cannot carry a photo, so when an
/// evidence image is available we use the OS share sheet (text + image);
/// the user picks X and taps Post themselves. If sharing fails, fall back
/// to the text-only intent URL.
class XShareService {
  Future<bool> shareWithEvidence({
    required String text,
    required String intentUrl,
    File? image,
  }) async {
    if (image != null) {
      try {
        final result = await Share.shareXFiles([XFile(image.path)], text: text);
        // Android < 14 reports `unavailable`; treat that as shared.
        return result.status != ShareResultStatus.dismissed;
      } catch (_) {
        // fall through to the text-only intent
      }
    }
    return openPrefilledPost(intentUrl);
  }

  Future<bool> openPrefilledPost(String intentUrl) async {
    final uri = Uri.parse(intentUrl);
    return launchUrl(uri, mode: LaunchMode.externalApplication);
  }
}
