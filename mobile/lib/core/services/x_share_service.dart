import 'package:url_launcher/url_launcher.dart';

/// FR-14/FR-15: open X with the report pre-filled and the authority
/// pre-tagged, but NEVER publish automatically -- the user must review and
/// submit the post themselves inside the X app/website.
class XShareService {
  Future<bool> openPrefilledPost(String intentUrl) async {
    final uri = Uri.parse(intentUrl);
    return launchUrl(uri, mode: LaunchMode.externalApplication);
  }
}
