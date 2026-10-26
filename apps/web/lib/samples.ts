/** Synthetic postings for the playground — written for this demo, not scraped. */
export interface Sample {
  id: string;
  label: string;
  text: string;
}

export const SAMPLES: Sample[] = [
  {
    id: "backend",
    label: "Senior backend (remote, salary)",
    text: `Senior Backend Engineer - Acme Cloud

We are hiring a Senior Backend Engineer to join our fully remote team.

Requirements:
- 5+ years of professional experience building web services
- Strong Python and SQL; hands-on experience with AWS and Docker
- Nice to have: Kubernetes, Terraform

Compensation: $120,000 - $150,000 per year, plus equity and a home-office stipend.`,
  },
  {
    id: "designer",
    label: "Product designer (hybrid, no salary)",
    text: `Product Designer
Northwind Studio - Manila, Philippines (Hybrid, 3 days in office)

You will own the end-to-end design of our scheduling product, from research to polished UI.

What you bring:
- 3-5 years of product design experience
- Fluent in Figma; comfortable running user interviews
- A portfolio that shows your process

We offer HMO, 15 days leave and a learning budget.`,
  },
  {
    id: "intern",
    label: "ML intern (onsite, hourly)",
    text: `Machine Learning Intern - Summer 2026

Join the applied research group at Lumen Labs in Austin, TX. This is an on-site internship.

You should be a current student with solid Python and PyTorch fundamentals. Experience with
Git is expected. Familiarity with Spark is a plus.

Pay: $38 - $45 per hour. No prior industry experience required.`,
  },
];
