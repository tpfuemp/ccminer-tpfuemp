// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Route handlers for the REST API — see docs/api-rest.md section 6.
 *
 * Each handler collects a snapshot (api_model.h), builds JSON from it and
 * returns an HTTP status. Handlers never touch the socket.
 *
 * Routes this build does not serve are registered available = false, so they
 * answer 501 and the capability list stays truthful.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "miner.h"
#include "algos.h"
#include "api_model.h"
#include "api_routes.h"

extern struct stratum_ctx stratum;

/* ---------------------------------------------------------------- helpers */

/* Every reply carries the miner envelope (docs/api-rest.md section 5). */
static json_t *envelope(void)
{
	json_t *root = json_object();
	if (!root) return NULL;
	json_t *m = api_build_miner_json();
	if (!m) { json_decref(root); return NULL; }
	json_object_set_new(root, "miner", m);
	return root;
}

static int oom(char *errmsg, size_t errlen)
{
	snprintf(errmsg, errlen, "out of memory");
	return 500;
}

/* ?id=N / ?index=N, or the trailing path segment of a prefix route. */
static bool query_int(const char *query, const char *key, long *out)
{
	size_t klen = strlen(key);
	for (const char *p = query; p && *p; ) {
		if (strncmp(p, key, klen) == 0 && p[klen] == '=') {
			char *end = NULL;
			long v = strtol(p + klen + 1, &end, 10);
			if (end == p + klen + 1) return false;
			*out = v;
			return true;
		}
		p = strchr(p, '&');
		if (p) p++;
	}
	return false;
}

static bool trailing_int(const char *path, long *out)
{
	const char *slash = strrchr(path, '/');
	if (!slash || !slash[1]) return false;
	char *end = NULL;
	long v = strtol(slash + 1, &end, 10);
	if (end == slash + 1 || *end) return false;
	*out = v;
	return true;
}

/* ---------------------------------------------------------------- handlers */

static int h_index(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);

	json_t *index = json_object();
	json_t *caps = json_array();
	json_t *links = json_object();
	if (!index || !caps || !links) {
		json_decref(root);
		if (index) json_decref(index);
		if (caps) json_decref(caps);
		if (links) json_decref(links);
		return oom(e, n);
	}

	/* Derived from the table, so it cannot claim a capability that is not
	 * routed, nor omit one that is. */
	size_t count = 0;
	const api_route *routes = api_routes_get(&count);
	for (size_t i = 0; i < count; i++) {
		if (!routes[i].available)
			continue;
		const char *p = routes[i].path;
		if (strcmp(p, "/metrics") == 0) {
			json_array_append_new(caps, json_string("metrics"));
			continue;
		}
		if (strncmp(p, "/api/v1/", 8) != 0)
			continue;
		const char *name = p + 8;
		if (!*name)
			continue;                       /* the index itself */
		char buf[64];
		snprintf(buf, sizeof(buf), "%s", name);
		size_t l = strlen(buf);
		if (l && buf[l-1] == '/') buf[l-1] = '\0';   /* prefix route */
		for (char *s = buf; *s; s++) if (*s == '/') *s = '.';  /* pools/switch -> pools.switch */
		/* de-duplicate: /pools and /pools/ both map to "pools" */
		bool seen = false;
		size_t na = json_array_size(caps);
		for (size_t k = 0; k < na; k++) {
			const char *ex = json_string_value(json_array_get(caps, k));
			if (ex && strcmp(ex, buf) == 0) { seen = true; break; }
		}
		if (!seen)
			json_array_append_new(caps, json_string(buf));
	}

	json_object_set_new(links, "summary", json_string("/api/v1/summary"));
	json_object_set_new(links, "devices", json_string("/api/v1/devices"));
	json_object_set_new(index, "capabilities", caps);
	json_object_set_new(index, "links", links);
	json_object_set_new(root, "index", index);
	*out = root;
	return 200;
}

static int h_summary(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	struct api_summary_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);

	api_collect_summary(&snap);
	json_t *o = api_build_summary_json(&snap);
	if (!o) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "summary", o);
	*out = root;
	return 200;
}

static int h_threads(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long want = -1;
	bool filtered = query_int(req->query, "id", &want);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	bool matched = false;
	for (int i = 0; i < opt_n_threads; i++) {
		if (filtered && i != (int) want) continue;
		struct api_thread_snapshot snap;
		api_collect_thread(i, &snap);
		if (!snap.valid) continue;
		matched = true;
		json_t *t = api_build_thread_json(&snap);
		if (t) json_array_append_new(arr, t);
	}
	if (filtered && !matched) {
		json_decref(root);
		json_decref(arr);
		snprintf(e, n, "no such thread");
		return 404;
	}
	json_object_set_new(root, "threads", arr);
	*out = root;
	return 200;
}

/* /devices and /devices/{id} share this: the hardware snapshot plus the
 * hashrate from the thread bound to that device. */
static bool device_json(int gpu_id, json_t **dst)
{
	struct api_gpuhw_snapshot hw;
	struct api_thread_snapshot th;
	api_collect_gpuhw(gpu_id, &hw);
	if (!hw.valid)
		return false;

	memset(&th, 0, sizeof(th));
	for (int i = 0; i < opt_n_threads; i++) {
		struct api_thread_snapshot t;
		api_collect_thread(i, &t);
		if (t.valid && t.gpu_id == gpu_id) { th = t; break; }
	}
	*dst = api_build_device_json(&hw, th.valid ? &th : NULL);
	return *dst != NULL;
}

static int h_devices(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < cuda_num_devices(); i++) {
		json_t *d = NULL;
		if (device_json(i, &d))
			json_array_append_new(arr, d);
	}
	json_object_set_new(root, "devices", arr);
	*out = root;
	return 200;
}

static int h_device(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long id = 0;
	if (!trailing_int(req->path, &id) || id < 0 || id >= cuda_num_devices()) {
		snprintf(e, n, "no such device");
		return 404;
	}
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *d = NULL;
	if (!device_json((int) id, &d)) {
		json_decref(root);
		snprintf(e, n, "no such device");
		return 404;
	}
	json_object_set_new(root, "device", d);
	*out = root;
	return 200;
}

static int h_system(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	struct api_system_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	api_collect_system(&snap);
	json_t *o = api_build_system_json(&snap);
	if (!o) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "system", o);
	*out = root;
	return 200;
}

static int h_pools(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long want = -1;
	bool filtered = query_int(req->query, "index", &want);

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	for (int i = 0; i < num_pools; i++) {
		if (filtered && i != (int) want) continue;
		struct api_pool_snapshot snap;
		api_collect_pool(i, &snap);
		json_t *p = api_build_pool_json(i, i == cur_pooln, &snap);
		if (p) json_array_append_new(arr, p);
	}
	if (filtered && json_array_size(arr) == 0) {
		json_decref(root);
		json_decref(arr);
		snprintf(e, n, "no such pool");
		return 404;
	}
	json_object_set_new(root, "pools", arr);
	*out = root;
	return 200;
}

static int h_pool(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	long idx = 0;
	if (!trailing_int(req->path, &idx) || idx < 0 || idx >= num_pools) {
		snprintf(e, n, "no such pool");
		return 404;
	}
	struct api_pool_snapshot snap;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	api_collect_pool((int) idx, &snap);
	json_t *p = api_build_pool_json((int) idx, (int) idx == cur_pooln, &snap);
	if (!p) { json_decref(root); return oom(e, n); }
	json_object_set_new(root, "pool", p);
	*out = root;
	return 200;
}

static int h_health(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *h = json_object();
	json_t *reasons = json_array();
	if (!h || !reasons) {
		json_decref(root);
		if (h) json_decref(h);
		if (reasons) json_decref(reasons);
		return oom(e, n);
	}

	const bool stratum_pool = (pools[cur_pooln].type & POOL_STRATUM) != 0;
	const bool connected = stratum_pool ? (stratum.curl != NULL) : true;
	const bool mining = opt_n_threads > 0 && !abort_flag;

	int devices_ok = 0;
	for (int i = 0; i < opt_n_threads; i++) {
		struct api_thread_snapshot t;
		api_collect_thread(i, &t);
		if (t.valid) devices_ok++;
	}

	if (!connected)
		json_array_append_new(reasons, json_string("pool_disconnected"));

	/* A deliberately stopped miner is healthy: a manager-initiated stop is not
	 * a fault (docs/api-rest.md section 6.6). */
	const bool degraded = !connected;

	json_object_set_new(h, "status", json_string(degraded ? "degraded" : "ok"));
	json_object_set_new(h, "mining", json_boolean(mining));
	json_object_set_new(h, "pool_connected", stratum_pool ? json_boolean(connected) : json_null());
	json_object_set_new(h, "devices_ok", json_integer(devices_ok));
	if (json_array_size(reasons))
		json_object_set_new(h, "reasons", reasons);
	else
		json_decref(reasons);

	json_object_set_new(root, "health", h);
	*out = root;
	return degraded ? 503 : 200;
}

static int h_algos(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *arr = json_array();
	if (!arr) { json_decref(root); return oom(e, n); }

	/* No algo advertises tunable parameters yet; an empty list is a true
	 * statement about this build. */
	for (int i = 0; i < ALGO_COUNT; i++) {
		if (!algo_names[i] || !*algo_names[i])
			continue;
		json_t *a = json_object();
		if (!a) continue;
		json_object_set_new(a, "name", json_string(algo_names[i]));
		json_object_set_new(a, "params", json_array());
		json_array_append_new(arr, a);
	}
	json_object_set_new(root, "algos", arr);
	*out = root;
	return 200;
}

static int h_config(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx;
	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *c = json_object();
	if (!c) { json_decref(root); return oom(e, n); }

	/* Credentials are masked: never return user or pass (section 6.8). */
	char algo[64] = { 0 };
	get_currentalgo(algo, sizeof(algo));
	json_object_set_new(c, "algo", json_string(algo));
	json_object_set_new(c, "threads", json_integer(opt_n_threads));
	json_object_set_new(c, "devices", json_integer(active_gpus));
	/* Intensity is per device, not a scalar: report what was configured for
	 * each, and null where it is left on auto. */
	json_t *inten = json_array();
	if (inten) {
		for (int i = 0; i < active_gpus && i < MAX_GPUS; i++)
			json_array_append_new(inten, gpus_intensity[i]
				? json_real(throughput2intensity(gpus_intensity[i]))
				: json_null());
		json_object_set_new(c, "intensity", inten);
	}
	json_object_set_new(c, "benchmark", json_boolean(opt_benchmark));
	json_object_set_new(c, "debug", json_boolean(opt_debug));
	json_object_set_new(c, "pools", json_integer(num_pools));
	json_object_set_new(root, "config", c);
	*out = root;
	return 200;
}

/* ------------------------------------------------------------ write routes */

/* Body parsing is explicit about types: a wrong type is 400, never a crash and
 * never a silent coercion. An absent body is treated as an empty object so a
 * verb that needs no arguments works with or without one. */
static json_t *parse_body(const api_request *req, char *e, size_t n, int *status)
{
	if (!req->body_len) {
		*status = 200;
		return json_object();
	}
	json_error_t err;
	json_t *root = json_loads(req->body, 0, &err);
	if (!root) {
		snprintf(e, n, "invalid JSON: %s", err.text);
		*status = 400;
		return NULL;
	}
	if (!json_is_object(root)) {
		json_decref(root);
		snprintf(e, n, "body must be a JSON object");
		*status = 400;
		return NULL;
	}
	*status = 200;
	return root;
}

static json_t *result_ok(void)
{
	json_t *root = envelope();
	if (!root) return NULL;
	json_t *res = json_object();
	if (!res) { json_decref(root); return NULL; }
	json_object_set_new(res, "ok", json_true());
	json_object_set_new(root, "result", res);
	return root;
}

static int h_pools_switch(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	json_t *jnext = json_object_get(body, "next");
	json_t *jindex = json_object_get(body, "index");

	if (jnext && jindex) {
		json_decref(body);
		snprintf(e, n, "send either index or next, not both");
		return 400;
	}
	if (!jnext && !jindex) {
		json_decref(body);
		snprintf(e, n, "index or next is required");
		return 400;
	}

	bool ok;
	if (jnext) {
		if (!json_is_boolean(jnext)) {
			json_decref(body);
			snprintf(e, n, "next must be a boolean");
			return 400;
		}
		if (!json_is_true(jnext)) {
			json_decref(body);
			snprintf(e, n, "next must be true");
			return 400;
		}
		ok = pool_switch_next(-1);
	} else {
		if (!json_is_integer(jindex)) {
			json_decref(body);
			snprintf(e, n, "index must be an integer");
			return 400;
		}
		json_int_t idx = json_integer_value(jindex);
		/* The binary command accepts a negative index and passes it straight
		 * through; the REST contract says an out-of-range index is 404. */
		if (idx < 0 || idx >= num_pools) {
			json_decref(body);
			snprintf(e, n, "no such pool");
			return 404;
		}
		ok = (idx == cur_pooln) ? true : pool_switch(-1, (int) idx);
	}
	json_decref(body);

	if (!ok) {
		snprintf(e, n, "pool switch failed");
		return 409;
	}

	json_t *root = envelope();
	if (!root) return oom(e, n);
	json_t *res = json_object();
	if (!res) { json_decref(root); return oom(e, n); }
	json_object_set_new(res, "ok", json_true());
	json_object_set_new(res, "active", json_integer(cur_pooln));
	json_object_set_new(root, "result", res);
	*out = root;
	return 200;
}

/* parse_arg('o') accepts exactly these, and calls show_usage_and_exit() on
 * anything else — which would terminate the miner from an API request. The
 * scheme is therefore validated here, before the value can reach it. */
static bool scheme_supported(const char *url)
{
	static const char *ok[] = {
		"http://", "https://", "stratum+tcp://", "stratum+ssl://", "stratum+tcps://"
	};
	for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++) {
		size_t l = strlen(ok[i]);
		if (strlen(url) > l && strncasecmp(url, ok[i], l) == 0)
			return true;
	}
	return false;
}

static int h_pools_url(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) ctx;
	int status = 200;
	json_t *body = parse_body(req, e, n, &status);
	if (!body) return status;

	json_t *jurl = json_object_get(body, "url");
	json_t *juser = json_object_get(body, "user");
	json_t *jpass = json_object_get(body, "pass");

	if (!jurl || !json_is_string(jurl)) {
		json_decref(body);
		snprintf(e, n, "url is required and must be a string");
		return 400;
	}
	if ((juser && !json_is_string(juser)) || (jpass && !json_is_string(jpass))) {
		json_decref(body);
		snprintf(e, n, "user and pass must be strings");
		return 400;
	}

	const char *url = json_string_value(jurl);
	const char *user = juser ? json_string_value(juser) : NULL;
	const char *pass = jpass ? json_string_value(jpass) : NULL;

	if (!scheme_supported(url)) {
		json_decref(body);
		snprintf(e, n, "unsupported url scheme");
		return 400;
	}

	/* Assemble the scheme://user:pass@host form parse_arg('o') expects, so a
	 * client never has to build a packed string (docs/api-rest.md 6.10). The
	 * credentials are stripped back out of the stored url by the parser, which
	 * is why /pools can never return them. */
	char packed[1280];
	int wrote;
	if (user && *user) {
		const char *sep = strstr(url, "://");
		size_t schemelen = (size_t) (sep - url) + 3;
		if (schemelen >= sizeof(packed)) {
			json_decref(body);
			snprintf(e, n, "url too long");
			return 400;
		}
		wrote = snprintf(packed, sizeof(packed), "%.*s%s:%s@%s",
			(int) schemelen, url, user, pass ? pass : "x", url + schemelen);
	} else {
		wrote = snprintf(packed, sizeof(packed), "%s", url);
	}
	json_decref(body);

	if (wrote < 0 || (size_t) wrote >= sizeof(packed)) {
		snprintf(e, n, "url too long");
		return 400;
	}

	if (!pool_switch_url(packed)) {
		snprintf(e, n, "pool url change failed");
		return 409;
	}

	json_t *root = result_ok();
	if (!root) return oom(e, n);
	*out = root;
	return 200;
}

static int h_quit(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req;
	json_t *root = result_ok();
	if (!root) return oom(e, n);

	/* Only flag it. The caller shuts down after this response has been sent,
	 * so the client always sees its 200. */
	if (ctx)
		((api_ctx *) ctx)->quit_requested = true;

	*out = root;
	return 200;
}

/* Registered so the route exists and answers 501 rather than 404. */
static int h_unavailable(const api_request *req, void *ctx, json_t **out, char *e, size_t n)
{
	(void) req; (void) ctx; (void) out;
	snprintf(e, n, "not implemented yet");
	return 501;
}

/* ------------------------------------------------------------------- table */

static const api_route g_routes[] = {
	{ API_M_GET,  "/api/v1",              API_PRIV_READ,    true,  h_index    },
	{ API_M_GET,  "/api/v1/summary",      API_PRIV_READ,    true,  h_summary  },
	{ API_M_GET,  "/api/v1/threads",      API_PRIV_READ,    true,  h_threads  },
	{ API_M_GET,  "/api/v1/devices",      API_PRIV_READ,    true,  h_devices  },
	{ API_M_GET,  "/api/v1/devices/",     API_PRIV_READ,    true,  h_device   },
	{ API_M_GET,  "/api/v1/system",       API_PRIV_READ,    true,  h_system   },
	{ API_M_GET,  "/api/v1/pools",        API_PRIV_READ,    true,  h_pools    },
	{ API_M_GET,  "/api/v1/pools/",       API_PRIV_READ,    true,  h_pool     },
	{ API_M_GET,  "/api/v1/health",       API_PRIV_READ,    true,  h_health   },
	{ API_M_GET,  "/api/v1/config",       API_PRIV_READ,    true,  h_config   },
	{ API_M_GET,  "/api/v1/algos",        API_PRIV_READ,    true,  h_algos    },

	/* served by the binary API, not yet ported to JSON */
	{ API_M_GET,  "/api/v1/history",      API_PRIV_READ,    false, h_unavailable },
	{ API_M_GET,  "/api/v1/scanlog",      API_PRIV_READ,    false, h_unavailable },
	{ API_M_GET,  "/api/v1/meminfo",      API_PRIV_READ,    false, h_unavailable },

	/* write routes */
	{ API_M_POST, "/api/v1/pools/switch", API_PRIV_WRITE,   true,  h_pools_switch },
	{ API_M_POST, "/api/v1/pools/url",    API_PRIV_WRITE,   true,  h_pools_url    },
	{ API_M_POST, "/api/v1/quit",         API_PRIV_WRITE,   true,  h_quit         },

	/* control routes, not implemented yet */
	{ API_M_GET,  "/api/v1/control/state",   API_PRIV_READ,    false, h_unavailable },
	{ API_M_POST, "/api/v1/control/start",   API_PRIV_CONTROL, false, h_unavailable },
	{ API_M_POST, "/api/v1/control/pause",   API_PRIV_CONTROL, false, h_unavailable },
	{ API_M_POST, "/api/v1/control/stop",    API_PRIV_CONTROL, false, h_unavailable },
	{ API_M_POST, "/api/v1/control/profile", API_PRIV_CONTROL, false, h_unavailable },

	/* Prometheus metrics, not implemented yet. Outside /api/v1 by design. */
	{ API_M_GET,  "/metrics",               API_PRIV_READ,    false, h_unavailable },
};

const api_route *api_routes_get(size_t *count)
{
	*count = sizeof(g_routes) / sizeof(g_routes[0]);
	return g_routes;
}

char *api_routes_miner_json_str(void)
{
	json_t *m = api_build_miner_json();
	if (!m) return NULL;
	char *s = json_dumps(m, JSON_COMPACT);
	json_decref(m);
	return s;
}
