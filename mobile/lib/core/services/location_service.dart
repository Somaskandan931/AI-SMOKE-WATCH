import 'dart:convert';

import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;

/// FR-11/FR-12 + PRD section 9: capture location + timestamp, with
/// permission, and show something a human recognizes -- not raw
/// coordinates.
///
/// Getting the coordinates costs nothing and needs no API key
/// (`geolocator`, using the phone's own GPS/network location).
///
/// Turning those coordinates into a landmark/place name ("Anna Salai,
/// Chennai" instead of "13.05974, 80.24714") is reverse geocoding, and you
/// have two no-friction options:
///
/// 1. OpenStreetMap Nominatim (used below) -- completely free, no API key,
///    no billing account. Good enough for a hackathon demo. Please respect
///    their usage policy (max ~1 request/sec, set a real User-Agent) --
///    https://operations.osmfoundation.org/policies/nominatim/
///
/// 2. Google Maps Geocoding API -- needs an API key + billing enabled on a
///    Google Cloud project, but returns richer, more consistently-named
///    results (e.g. actual POI/landmark names) in areas where OSM data is
///    thin. Swap `reverseGeocode()`'s body for a call to
///    `https://maps.googleapis.com/maps/api/geocode/json?latlng=$lat,$lng&key=$apiKey`
///    and read `results[0].formatted_address` if you get a key later.
class LocationService {
  static const String _nominatimUrl = 'https://nominatim.openstreetmap.org/reverse';

  /// Requests permission (asking only if not already granted/denied
  /// permanently) and returns the current position, or null if permission
  /// was refused. Never collects location without permission (PRD section 9
  /// / FR-11).
  Future<Position?> getCurrentPosition() async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return null;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return null;
    }
    if (permission == LocationPermission.deniedForever) return null;

    return Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );
  }

  /// Turns coordinates into a short, human-readable place/landmark string,
  /// e.g. "Anna Salai, Thousand Lights, Chennai" -- this is the "actual
  /// place or landmark" shown on the Location Confirmation and Report
  /// Preview screens instead of raw GPS numbers.
  ///
  /// Falls back to the raw "lat, lng" string if reverse geocoding fails
  /// (offline, rate-limited, etc.) so the flow never gets stuck.
  Future<String> reverseGeocode(double latitude, double longitude) async {
    final uri = Uri.parse(_nominatimUrl).replace(queryParameters: {
      'lat': latitude.toString(),
      'lon': longitude.toString(),
      'format': 'jsonv2',
      'zoom': '18', // building/POI level detail, not just city
      'addressdetails': '1',
    });

    try {
      final response = await http.get(
        uri,
        // Nominatim's usage policy requires a descriptive User-Agent.
        headers: {'User-Agent': 'SmokeWatchApp/0.1 (civic-reporting-hackathon)'},
      ).timeout(const Duration(seconds: 8));

      if (response.statusCode != 200) {
        return _fallback(latitude, longitude);
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final address = data['address'] as Map<String, dynamic>?;

      // Prefer a named landmark/road over the full postal address.
      final landmark = address?['amenity'] ??
          address?['building'] ??
          address?['road'] ??
          address?['neighbourhood'];
      final locality = address?['suburb'] ?? address?['city_district'] ?? address?['city'];

      final parts = [landmark, locality].whereType<String>().toSet().toList();
      if (parts.isNotEmpty) return parts.join(', ');

      // Nominatim's own best-guess full name, as a second fallback.
      final displayName = data['display_name'] as String?;
      if (displayName != null && displayName.isNotEmpty) {
        return displayName.split(',').take(2).join(',').trim();
      }

      return _fallback(latitude, longitude);
    } catch (_) {
      return _fallback(latitude, longitude);
    }
  }

  String _fallback(double latitude, double longitude) =>
      '${latitude.toStringAsFixed(5)}, ${longitude.toStringAsFixed(5)}';
}
