/* *
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* */

/**
 * ============================================================================
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "string_hash.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>

#ifdef FARMHASH_BIG_ENDIAN
#define uint32_in_expected_order(x) (BSwap32_(x))
#define uint64_in_expected_order(x) (BSwap64_(x))
#else
#define uint32_in_expected_order(x) (x)
#define uint64_in_expected_order(x) (x)
#endif

using namespace std;


// Some primes between 2^63 and 2^64 for various uses.
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

// Magic numbers for 32-bit hashing.  Copied from Murmur3.
static const uint32_t c1 = 0xcc9e2d51;
static const uint32_t c2 = 0x1b873593;

unsigned long Rotr64 (unsigned long val, int shift)
{
    shift &= 0x3f;
    val = (val<<(0x40 - shift)) | (val >> shift);
    return val;
}

uint64_t BasicRotate64(uint64_t val, int shift) 
{
  // Avoid shifting by 64: doing so yields an undefined result.
  return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

uint32_t Fetch32(const char *p) 
{
  uint32_t result;
  memcpy(&result, p, sizeof(result));
  return uint32_in_expected_order(result);
}

uint64_t Fetch64(const char *p) 
{
    uint64_t result;
    memcpy(&result, p, sizeof(result));
    return uint64_in_expected_order(result);
}

uint64_t Rotate64(uint64_t val, int shift) 
{
    return sizeof(unsigned long) == sizeof(val) ?
      Rotr64(val, shift) :
      BasicRotate64(val, shift);
}

uint64_t ShiftMix(uint64_t val)
{
    return val ^ (val >> 47);
}

uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) 
{
    // Murmur-inspired hashing.
    uint64_t a = (u ^ v) * mul;
    a ^= (a >> 47);
    uint64_t b = (v ^ a) * mul;
    b ^= (b >> 47);
    b *= mul;
    return b;
}


uint64_t HashLen0to16(const char *s, size_t len) 
{
    if (len >= 8) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = Fetch64(s) + k2;
        uint64_t b = Fetch64(s + len - 8);
        uint64_t c = Rotate64(b, 37) * mul + a;
        uint64_t d = (Rotate64(a, 25) + b) * mul;
        return HashLen16(c, d, mul);
    }
    if (len >= 4) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = Fetch32(s);
        return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
    }
    if (len > 0) {
        uint8_t a = s[0];
        uint8_t b = s[len >> 1];
        uint8_t c = s[len - 1];
        uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
        uint32_t z = len + (static_cast<uint32_t>(c) << 2);
        return ShiftMix(y * k2 ^ z * k0) * k2;
    }
    return k2;
}

uint64_t HashLen17to32(const char *s, size_t len) 
{
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch64(s) * k1;
    uint64_t b = Fetch64(s + 8);
    uint64_t c = Fetch64(s + len - 8) * mul;
    uint64_t d = Fetch64(s + len - 16) * k2;
    return HashLen16(Rotate64(a + b, 43) + Rotate64(c, 30) + d,
                   a + Rotate64(b + k2, 18) + c, mul);
}

uint64_t HashLen33to64(const char *s, size_t len)
{
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch64(s) * k2;
    uint64_t b = Fetch64(s + 8);
    uint64_t c = Fetch64(s + len - 8) * mul;
    uint64_t d = Fetch64(s + len - 16) * k2;
    uint64_t y = Rotate64(a + b, 43) + Rotate64(c, 30) + d;
    uint64_t z = HashLen16(y, a + Rotate64(b + k2, 18) + c, mul);
    uint64_t e = Fetch64(s + 16) * mul;
    uint64_t f = Fetch64(s + 24);
    uint64_t g = (y + Fetch64(s + len - 32)) * mul;
    uint64_t h = (z + Fetch64(s + len - 24)) * mul;
    return HashLen16(Rotate64(e + f, 43) + Rotate64(g, 30) + h,
                   e + Rotate64(f + a, 18) + g, mul);
}

pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) 
{
    a += w;
    b = Rotate64(b + a + z, 21);
    uint64_t c = a;
    a += x;
    a += y;
    b += Rotate64(a, 44);
    return make_pair(a + z, b + c);
}

// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
pair<uint64_t, uint64_t> WeakHashLen32WithSeeds0(const char* s, uint64_t a, uint64_t b) 
{
    return WeakHashLen32WithSeeds(Fetch64(s),
                                Fetch64(s + 8),
                                Fetch64(s + 16),
                                Fetch64(s + 24),
                                a,
                                b);
}         

uint64_t Hash64(const char *s, size_t len) 
{
    const uint64_t seed = 81;
    if (len <= 32) {
        if (len <= 16) {
            return HashLen0to16(s, len);
        } else {
            return HashLen17to32(s, len);
        }
    } else if (len <= 64) {
        return HashLen33to64(s, len);
    }

    // For strings over 64 bytes we loop.  Internal state consists of
    // 56 bytes: v, w, x, y, and z.
    uint64_t x = seed;
    uint64_t y = seed * k1 + 113;
    uint64_t z = ShiftMix(y * k2 + 113) * k2;
    pair<uint64_t, uint64_t> v = make_pair(0, 0);
    pair<uint64_t, uint64_t> w = make_pair(0, 0);
    x = x * k2 + Fetch64(s);

    // Set end so that after the loop we have 1 to 64 bytes left to process.
    const char* end = s + ((len - 1) / 64) * 64;
    const char* last64 = end + ((len - 1) & 63) - 63;
    assert(s + len - 64 == last64);
    do {
    x = Rotate64(x + y + v.first + Fetch64(s + 8), 37) * k1;
    y = Rotate64(y + v.second + Fetch64(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch64(s + 40);
    z = Rotate64(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds0(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds0(s + 32, z + w.second, y + Fetch64(s + 16));
    std::swap(z, x);
    s += 64;
    } while (s != end);
    uint64_t mul = k1 + ((z & 0xff) << 1);
    // Make s point to the last 64 bytes of input.
    s = last64;
    w.first += ((len - 1) & 63);
    v.first += w.first;
    w.first += v.first;
    x = Rotate64(x + y + v.first + Fetch64(s + 8), 37) * mul;
    y = Rotate64(y + v.second + Fetch64(s + 48), 42) * mul;
    x ^= w.second * 9;
    y += v.first * 9 + Fetch64(s + 40);
    z = Rotate64(z + w.first, 33) * mul;
    v = WeakHashLen32WithSeeds0(s, v.second * mul, x + w.first);
    w = WeakHashLen32WithSeeds0(s + 32, z + w.second, y + Fetch64(s + 16));
    std::swap(z, x);
    return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z,
                   HashLen16(v.second, w.second, mul) + x,
                   mul);
}

uint64_t FingerprintCat64(const uint64_t fp1, const uint64_t fp2) 
{
    static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
    uint64_t result = fp1 ^ kMul;
    result ^= ShiftMix(fp2 * kMul) * kMul;
    result *= kMul;
    result = ShiftMix(result) * kMul;
    result = ShiftMix(result);
    return result;
}

uint64_t Fingerprint64(string ss) 
{
    const char* s = ss.c_str();
    uint64_t len = ss.length();
    return Hash64(s, len);
}
