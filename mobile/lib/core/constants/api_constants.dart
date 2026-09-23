/// Backend API configuration.
///
/// For the hackathon demo (Android-only build):
/// - Android emulator: 10.0.2.2 reaches your host machine's localhost
///   (this is the default below).
/// - Physical Android device on the same Wi-Fi: use your machine's LAN IP.
/// - Change this single constant rather than hunting through the codebase.
class ApiConstants {
  ApiConstants._();

  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000/api',
  );

  static const String health = '$baseUrl/health';
  static const String detect = '$baseUrl/detect';
  static const String plateDetect = '$baseUrl/plate/detect';
  static const String plateOcr = '$baseUrl/plate/ocr';
  static const String reportGenerate = '$baseUrl/report/generate';

  /// Run-time override, e.g. from a settings screen, so the demo can be
  /// pointed at a different backend host without a rebuild.
  static String? overrideBaseUrl;

  static String resolve(String path) => overrideBaseUrl ?? path;
}
