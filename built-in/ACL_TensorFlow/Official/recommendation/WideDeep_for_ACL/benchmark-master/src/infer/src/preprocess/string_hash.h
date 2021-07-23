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

#ifndef STRING_HASH_H
#define STRING_HASH_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <utility>
#include <iostream>
#include <fstream>

using namespace std;

unsigned RotL(unsigned val,int shift);
unsigned long RrotL(unsigned long val,int shift);
unsigned RotR(unsigned val,int shift);
unsigned long LrotR(unsigned long val,int shift);

uint64_t BasicRotate64(uint64_t val, int shift);
uint32_t Fetch32(const char *p);
uint64_t Fetch64(const char *p);
uint64_t Rotate64(uint64_t val, int shift);
uint64_t ShiftMix(uint64_t val);
uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul);
uint64_t HashLen0to16(const char *s, size_t len);
uint64_t HashLen17to32(const char *s, size_t len);
uint64_t HashLen33to64(const char *s, size_t len);

pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b);
pair<uint64_t, uint64_t> WeakHashLen32WithSeeds0(
    const char* s, uint64_t a, uint64_t b);
    
uint64_t Hash64(const char *s, size_t len);
uint64_t FingerprintCat64(const uint64_t fp1, const uint64_t fp2);
uint64_t Fingerprint64(string ss);

#endif

