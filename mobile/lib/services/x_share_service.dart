import 'package:url_launcher/url_launcher.dart';

/// FR-14 / FR-15: opens X (Twitter) with the pre-filled report and
/// relevant-authority mention, but never auto-publishes — the user still
/// has to hit "Post" inside X themselves.
class XShareService {
  Future<bool> openIntent(String intentUrl) async {
    final uri = Uri.parse(intentUrl);
    if (await canLaunchUrl(uri)) {
      return launchUrl(uri, mode: LaunchMode.externalApplication);
    }
    return false;
  }
}
