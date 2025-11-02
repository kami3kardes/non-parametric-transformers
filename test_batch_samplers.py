"""Test script for all batch sampling methods."""
import numpy as np
import torch
import sys

print("=" * 80)
print("Testing NPT Batch Samplers")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    from npt.utils.batch_utils import (
        StratifiedIndexSampler,
        ClusteredIndexSampler,
        PrototypeIndexSampler,
        LearnedPrototypeIndexSampler,
        collate_with_pre_batching
    )
    from npt.prototypes import LearnedPrototypes
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test data
np.random.seed(42)
n_samples = 100
n_features = 10
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, 3, n_samples)  # 3 classes
clusters = np.random.randint(0, 5, n_samples)  # 5 clusters
row_indices = np.arange(n_samples)

print(f"\nTest data: {n_samples} samples, {n_features} features, {len(np.unique(y))} classes")

# Test 1: StratifiedIndexSampler
print("\n2. Testing StratifiedIndexSampler...")
try:
    sampler = StratifiedIndexSampler(y=y, n_splits=5, shuffle=True, random_state=42)
    new_order, batch_sizes = sampler.get_stratified_test_array(row_indices)
    print(f"   ✓ StratifiedIndexSampler works")
    print(f"     - Returned {len(batch_sizes)} batches with sizes: {batch_sizes}")
    print(f"     - Total samples: {sum(batch_sizes)} (expected: {n_samples})")
    assert sum(batch_sizes) == n_samples, "Sample count mismatch!"
except Exception as e:
    print(f"   ✗ StratifiedIndexSampler failed: {e}")

# Test 2: ClusteredIndexSampler
print("\n3. Testing ClusteredIndexSampler...")
try:
    sampler = ClusteredIndexSampler(y=clusters, n_splits=5, shuffle=True, random_state=42)
    new_order, batch_sizes = sampler.get_stratified_test_array(row_indices)
    print(f"   ✓ ClusteredIndexSampler works")
    print(f"     - Returned {len(batch_sizes)} batches with sizes: {batch_sizes}")
    print(f"     - Total samples: {sum(batch_sizes)} (expected: {n_samples})")
    assert sum(batch_sizes) == n_samples, "Sample count mismatch!"
except Exception as e:
    print(f"   ✗ ClusteredIndexSampler failed: {e}")

# Test 3: PrototypeIndexSampler
print("\n4. Testing PrototypeIndexSampler...")
try:
    # Create prototype info
    n_prototypes = 10
    prototype_indices = np.random.choice(n_samples, n_prototypes, replace=False)
    neighbors = []
    for proto in prototype_indices:
        # Each prototype has 5 random neighbors
        neighs = np.random.choice(n_samples, 5, replace=False)
        neighbors.append(neighs)

    proto_info = {
        'prototype_indices': prototype_indices,
        'neighbors': neighbors
    }

    sampler = PrototypeIndexSampler(y=proto_info, n_splits=5, shuffle=True, random_state=42)
    new_order, batch_sizes = sampler.get_stratified_test_array(row_indices)
    print(f"   ✓ PrototypeIndexSampler works")
    print(f"     - Used {n_prototypes} prototypes")
    print(f"     - Returned {len(batch_sizes)} batches with sizes: {batch_sizes}")
    print(f"     - Total samples: {sum(batch_sizes)} (expected: {n_samples})")
    assert sum(batch_sizes) == n_samples, "Sample count mismatch!"
except Exception as e:
    print(f"   ✗ PrototypeIndexSampler failed: {e}")

# Test 4: LearnedPrototypeIndexSampler
print("\n5. Testing LearnedPrototypeIndexSampler...")
try:
    # Create learned prototypes
    n_prototypes = 8
    prototype_dim = n_features
    learned_protos = LearnedPrototypes(
        n_prototypes=n_prototypes,
        prototype_dim=prototype_dim,
        device='cpu'
    )

    # Initialize from data
    learned_protos.init_from_data(X, method='random', random_state=42)

    # Create sampler
    sampler = LearnedPrototypeIndexSampler(
        dataset_features=X,
        prototypes_getter=learned_protos,
        k_neighbors=8,
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Update sampler (compute nearest neighbors)
    sampler.update(epoch=0)

    # Get batches
    new_order, batch_sizes = sampler.get_stratified_test_array(row_indices)
    print(f"   ✓ LearnedPrototypeIndexSampler works")
    print(f"     - Used {n_prototypes} learned prototypes")
    print(f"     - k_neighbors: {sampler.k_neighbors}")
    print(f"     - Returned {len(batch_sizes)} batches with sizes: {batch_sizes}")
    print(f"     - Total samples: {sum(batch_sizes)} (expected: {n_samples})")
    assert sum(batch_sizes) == n_samples, "Sample count mismatch!"
except Exception as e:
    print(f"   ✗ LearnedPrototypeIndexSampler failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: LearnedPrototypes class
print("\n6. Testing LearnedPrototypes class...")
try:
    protos = LearnedPrototypes(n_prototypes=5, prototype_dim=10, device='cpu')

    # Test forward pass
    test_input = torch.randn(20, 10)
    p_z_x, logits = protos(test_input, temperature=1.0)
    print(f"   ✓ LearnedPrototypes forward pass works")
    print(f"     - Input shape: {test_input.shape}")
    print(f"     - p(z|x) shape: {p_z_x.shape}")
    print(f"     - Logits shape: {logits.shape}")

    # Test IB loss
    test_labels = torch.randint(0, 3, (20,))
    loss, info = protos.ib_loss(test_input, test_labels)
    print(f"   ✓ LearnedPrototypes IB loss works")
    print(f"     - Loss: {loss.item():.4f}")
    print(f"     - Info: {info}")

except Exception as e:
    print(f"   ✗ LearnedPrototypes failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Collate function
print("\n7. Testing collate_with_pre_batching...")
try:
    # Create batch with pre-batched data
    batch_data = {
        'data': torch.randn(10, 5),
        'labels': torch.randint(0, 2, (10,)),
        'indices': torch.arange(10)
    }
    batch = [batch_data]

    result = collate_with_pre_batching(batch)
    print(f"   ✓ collate_with_pre_batching works")
    print(f"     - Input batch length: {len(batch)}")
    print(f"     - Output keys: {result.keys()}")

except Exception as e:
    print(f"   ✗ collate_with_pre_batching failed: {e}")

print("\n" + "=" * 80)
print("All batch sampler tests completed!")
print("=" * 80)
