# Native platform setup (Android)

This project targets **Android only**. The `android/` platform folder
already exists in this repo (generated via `flutter create .`) and its
manifest already has every permission the app's plugins need — there
is nothing left to retrofit.

## What's already done

`android/app/src/main/AndroidManifest.xml` already declares:

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />

<!-- Android 13+ (API 33+) photo picker permission -->
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<!-- Android 12 and below fallback -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
```

These cover every plugin the app uses that needs a native declaration:
`geolocator` (FR-11 location), `image_picker` (FR-01/FR-02 camera +
gallery), and `url_launcher` (FR-14 opening X — no extra manifest entry
needed on Android).

## 1. Point the app at your backend

`lib/core/constants/api_constants.dart` sets `baseUrl`:

- Android emulator -> `http://10.0.2.2:8000/api` (already the default)
- Physical Android device on the same Wi-Fi -> `http://<your-machine-LAN-IP>:8000/api`

## 2. Verify the build

This sandbox has no Flutter SDK, so these commands still need to be run
on your machine as the final word (see root `STATUS.md`):

```bash
cd mobile
flutter pub get
flutter analyze
flutter run
```
