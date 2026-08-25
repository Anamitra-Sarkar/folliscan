import { NextResponse, type NextRequest } from "next/server";

/** Route guard: protected pages require the session cookie (set on Firebase
 * sign-in). The cookie is a UX gate only — all real authorization happens
 * server-side via Firebase ID-token verification in the APIs. */
const PROTECTED = ["/dashboard", "/history"];

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const hasSession = req.cookies.get("folliscan_session")?.value;

  if (PROTECTED.some((p) => pathname.startsWith(p)) && !hasSession) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }
  if (pathname === "/login" && hasSession) {
    const url = req.nextUrl.clone();
    url.pathname = "/dashboard";
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/history/:path*", "/login"],
};
