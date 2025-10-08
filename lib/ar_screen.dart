import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

class ARScreen extends StatefulWidget {
  const ARScreen({super.key});

  @override
  State<ARScreen> createState() => _ARScreenState();
}

class _ARScreenState extends State<ARScreen> {
  static const String _viewType = 'dino_ar_view';
  static const MethodChannel _channel = MethodChannel('dino_ar_channel');
  bool _isCreatingPrototype = false;
  Timer? _placementTimer;

  @override
  void dispose() {
    _placementTimer?.cancel();
    super.dispose();
  }

  // Sends the reference image (prototype) to the native code
  Future<void> _createReferencePrototype() async {
    if (_isCreatingPrototype) return;
    setState(() => _isCreatingPrototype = true);

    try {
      await _channel.invokeMethod('clearHistory');
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) return;

      if (!mounted) return;

      final Uint8List fileBytes = await pickedFile.readAsBytes();
      // Call the native method to create the prototype
      final bool success = await _channel.invokeMethod('createPrototype', {
        'bytes': fileBytes,
      });
      if (!mounted) return;

      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✅ Prototype created! Starting segmentation...'),
          ),
        );

        // This enables segmentation on the native side.
        await _channel.invokeMethod('toggleSegmentation');

        _startPlacementAttempts();
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('❌ Could not create prototype.'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      debugPrint("Failed to create prototype: $e");
    } finally {
      if (mounted) {
        setState(() => _isCreatingPrototype = false);
      }
    }
  }

  // Starts a timer that repeatedly tries to place the AR object.
  void _startPlacementAttempts() {
    _placementTimer?.cancel();
    _placementTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      _placeObjectAtCentroid();
    });
  }

  // Attempts to place the object and stops segmentation on success.
  Future<void> _placeObjectAtCentroid() async {
    try {
      final bool success = await _channel.invokeMethod('placeObjectAtCentroid');
      if (!mounted) return;
      if (success) {
        _placementTimer?.cancel(); // Stop trying to place the object.
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✅ Object placed! Segmentation stopped.'),
          ),
        );

        // This disables segmentation on the native side, stopping inference.
        await _channel.invokeMethod('toggleSegmentation');
      }
    } catch (e) {
      debugPrint("Failed to place object: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('DINOv3 AR Segmentation'),
        backgroundColor: Colors.blue.withValues(alpha: 0.5),
      ),
      body: AndroidView(
        viewType: _viewType,
        layoutDirection: TextDirection.ltr,
        creationParamsCodec: const StandardMessageCodec(),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _isCreatingPrototype ? null : _createReferencePrototype,
        tooltip: 'Set Reference',
        child: _isCreatingPrototype
            ? const CircularProgressIndicator(color: Colors.white)
            : const Icon(Icons.upload_file),
      ),
    );
  }
}
